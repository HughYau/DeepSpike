import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.preprocessing as sp
import spikeinterface.widgets as sw
from dataset import spikeData
from preprocess import *
from models import AE, VAE
from clustering import *


def encode_func(config,threshold_s,device):

    if os.path.exists(config['data'].split('.')[0]+'.pt') and not config['reprocess']:
        # print('Using existing preprocessed data!!!')
        sig, events, eventIdx, eventSeq, fs = torch.load(config['data'].split('.')[0]+'.pt')
    else:
        print('Loading and preprocessing', config['data'].split('/')[-1])
        sig, events, eventIdx, eventSeq, fs = preprocess(config['data'],
                                                        config['cutoff'], config['poff'], config['noff'], config['denoise'])
        torch.save((sig, events, eventIdx, eventSeq, fs), config['data'].split('.')[0]+'.pt')
    # manual_res = pd.read_csv(config['res'],delimiter=',',skiprows=2,usecols=[0,2],names = ['time','cluster'],header=None)
    # manual_res['time_frames'] = round(manual_res['time'] * fs)
    # manual_res = manual_res[manual_res['cluster'] != 0]
    # print('ground truth labels:', len(labels_data[1]))
    # print('nlabels:', len(labels_data))


    flattened_events = events.flatten()
    median_events = torch.median(flattened_events)
    abs_deviation = torch.abs(events.flatten() - median_events)
    mad = abs_deviation.median()
    sigma = 1.4826*mad
    # print("Median:",median_events, "MAD:",mad,'Sigma',sigma)
    # print('Done!')
    if config['thresh'] > 0:
        thresholds = [config['thresh']]
    else:
        # thresholds = np.arange(0,1.0,0.025)#[0.6, 0.5, 0.4, 0.35, 0.3]#, 0.58, 0.57, 0.56, 0.55]
        thresholds = np.arange(median_events,1,threshold_s*sigma)

    all_emb = []
    all_evId = []
    all_xdata = []
    model_name = config['model']
    if model_name == None:
        model_name = 'models/model_H_'+repr(config['hidden'])+'L_'+repr(config['latent'])+'_T'+repr(config['thresh'])[2:]+'_CAE'+'.pt' 

    ### Obtain a distribution of number of events for different thresholds
    num_events = []

    for t in thresholds:
        posEventIdx = thresholdEvents(events,t)
        num_events.append(len(eventIdx[posEventIdx]))
        # print("Found %d events at %.4f threshold"%(len(eventIdx[posEventIdx]),t))
        
    num_events = np.array(num_events)
    thresholds = thresholds[num_events > 0]
    thresholds = thresholds[np.where(np.diff(num_events) < 0)[0][1]:]
    # print(thresholds)
    thresholds = thresholds[:len(thresholds)//2+1][::-1]
    # print('ground truth labels:', len(manual_res),'detected',len(thresholdEvents(events,thresholds[-1])))
    # gt_list.append(len(labels_data[1]))
    # detected_list.append(len(thresholdEvents(events,thresholds[-1])))

    evClsLabel = np.zeros(len(eventIdx),dtype=int)
    tIdx = 1
    threshIdx = 0
    tmp = 0

    for thresh in thresholds:
        # print("################## Using Threshold=%.2f ##############"%thresh)
        ### Make torch dataset
        dataset = spikeData(data=events, mask=config['mask'],event_index=eventSeq,\
                jigsaw=config['jigsaw'],shuffle=config['shuffle'],evMask=evClsLabel,thresh=thresh)

        #### Make training, validation and test sets
        N = len(dataset)
        if N == 0:
            print('Zero events found at %.2f threshold'%thresh)
            break
        if N > 100:
            
            B = config['batch']

            ### Instantiate a model! 
            nIp = dataset[0][0].shape[-1]
            if 'VAE' in model_name:
                model = VAE(nIp=nIp,nhid=config['hidden'],latent_dim=config['latent'])
            else:
                model = AE(nIp=nIp,nhid=config['hidden'],latent_dim=config['latent'])
            #criterion = nn.L1Loss() ############ Loss function to be optimized. 
            
            #pdb.set_trace()
            ### Load pretrained model

            
            model.load_state_dict(torch.load(config['model'],map_location=device))


            for loader in [DataLoader(dataset,batch_size=B, shuffle=False)]:
                if 'VAE' in model_name:
                    embeddings, xData, eventId = reduce_dimension_VAE(model,loader,device)
                else:
                    embeddings, xData, eventId = reduce_dimension(model,loader,device)
                tmp += len(embeddings)

                ### Perform clustering
                cls_labels, clusters, counts = make_clusters(embeddings,method='kmeans',nClus=2)
                cls_labels += tIdx
                clusters += tIdx
                ### Update event cluster label
                evClsLabel[eventId] = cls_labels

                ### Set the minority class to zero
                evClsLabel[evClsLabel == clusters[-1]] = 0

                ### Save embeddings for final clustering excluding the minority cluster
                if thresh == thresholds[-1]: # Save minority cluster for last thresh.
                    all_emb.append(embeddings)
                    all_evId.append(eventId)
                    all_xdata.append(xData)   
                else:

                #    pdb.set_trace()
                    all_emb.append(embeddings[cls_labels != clusters[-1]])
                    all_evId.append(eventId[cls_labels != clusters[-1]])
                    all_xdata.append(xData[cls_labels != clusters[-1]])   

                ### Remove detected spikes
                ### by setting them to zero
                # print(len(events[evClsLabel > 0]))
                events[evClsLabel > 0] = 0

                tIdx += len(clusters) 

        threshIdx += 1
    all_emb = torch.cat(all_emb, axis=0)
    all_xdata = np.concatenate(all_xdata, axis=0)
    all_evId = np.concatenate(all_evId,axis=0)
    all_evSeq = eventIdx[all_evId]
    
    return all_emb, all_xdata, all_evSeq,fs

# def cluster_func(config, all_emb, max_cluster, clustermethod, clmethod):
    

#     opFile = config['data'].split('.')[0] 
#     if clustermethod is not None:
#         opFile += '_'+config['model'].split('/')[-1].split('.')[0]+'_'+clustermethod+'_'+clmethod+'_results'
#     else:
#         opFile += '_'+config['model'].split('/')[-1].split('.')[0]+'_'+clmethod+'_results'
#     ### Make hierarchical clustering with fixed k
#     if config['clusters'] > 0:
#         cls_labels,clusters,counts = make_clusters(all_emb,nClus= max_cluster,\
#                 method=clmethod,automated = clustermethod)
#         # evClsLabel, clusters, counts, ordClsLabels, embeddings = draw_clustering_multi(all_xdata,\
#         #         all_emb,cls_labels,methods = vismethod,draw=True,filename=opFile)

#     return cls_labels
    ### Template matching of clusters
    
def output_df(config,fs, embeddings, all_xdata, all_evSeq, cls_labels):
    nClus = np.unique(cls_labels)
    tmplt = np.zeros((len(nClus),config['noff']+config['poff']))
    cIdx = 0

    for c in nClus:
        locData = all_xdata[cls_labels == c]
        if len(locData) > 10:
            tmplt[cIdx] = locData[np.random.permutation(len(locData))[:int(0.1*len(locData))]].mean(0)
        else:
            tmplt[cIdx] = locData.mean(0)
        cIdx += 1


  

    tmplt = tmplt[:,tmplt.shape[0]//4:-tmplt.shape[0]//4]

    ### Save final clustering in dataframe
    df = pd.DataFrame(columns=['peak_id','emb_x','emb_y','cluster'],index=np.arange(len(all_evSeq)))

    # print('Saving results to '+opFile+'.csv')
    embeddings = embeddings[all_evSeq.sort()[1]]
    cls_labels = cls_labels[all_evSeq.sort()[1]]

    df['cluster'] = cls_labels
    df['emb_x'] = embeddings[:,0]
    df['emb_y'] = embeddings[:,1]
    df['peak_id'] = all_evSeq.sort()[0]
    df['time_ms'] = df['peak_id']/fs*1000 # spike time in ms
    # if savecsv == True:
    #     df.to_csv(opFile+'.csv',index=False,float_format='%.4f')
    return df
    
        
        
def compare_sorters_vivo(config, df):
        recording = se.read_tdt(config['data'],stream_name="b'Wav1'")
        fs = recording.get_sampling_frequency()
        manual_res = pd.read_csv(config['res'],delimiter=',',skiprows=2,usecols=[0,2],names = ['time','cluster'],header=None)
        manual_res['time_frames'] = round(manual_res['time'] * fs)
        manual_res = manual_res[manual_res['cluster'] != 0]
        print(recording)
        print(f"Number of channels = {len(recording.get_channel_ids())}")
        print(f"Sampling frequency = {recording.get_sampling_frequency()} Hz")
        print(f"Number of segments= {recording.get_num_segments()}")
        print(f"Number of timepoints in seg0= {recording.get_num_frames(segment_index=0)}")
        from probeinterface import generate_linear_probe
        from probeinterface.plotting import plot_probe

        probe = generate_linear_probe(num_elec=1, ypitch=20, contact_shapes='circle', contact_shape_params={'radius': 6})

        # the probe has to be wired to the recording
        probe.set_device_channel_indices(np.arange(1))

        recording = recording.set_probe(probe)
        # plot_probe(probe)
        recording_f = sp.bandpass_filter(recording, freq_min=300, freq_max=5000)
        sorting_true = si.NumpySorting.from_times_labels(manual_res['time_frames'].values,manual_res['cluster'].values,fs)
        sorting_true.register_recording(recording_f)
        sorting_deep = si.NumpySorting.from_times_labels(df['peak_id'].values,df['cluster'].values.astype(int),fs)
        sorting_deep.register_recording(recording_f)

        comp = sc.compare_sorter_to_ground_truth(sorting_true, sorting_deep, exhaustive_gt=True, delta_time=10,match_mode = 'best')
        sw.plot_agreement_matrix(comp, ordered=True,backend='matplotlib',figtitle =  'Agreement matrix')
        perf = comp.get_performance()
        # plt.savefig(data_file[:-4]+f'\\agreement_{sorter_name}.png')
        # perf.to_csv(data_file[:-4]+f'\\performance_{sorter_name}.csv')


        try: 
            conf = comp.get_confusion_matrix()
            sw.plot_confusion_matrix(comp,backend='matplotlib',figtitle =  'Confusion matrix')
            # plt.savefig(data_file[:-4]+f'\\confusion_{sorter_name}.png')
            # conf.to_csv(data_file[:-4]+f'\\confusion_{sorter_name}.csv')
            
        except:
            print('confusion matrix failed')
        finally:
            comp.print_performance()
            # # print('well_detected_units',comp.get_well_detected_units(well_detected_score=0.8))
            # print('false_positive_units',comp.get_false_positive_units(redundant_score=0.2))
            comp.print_summary()
            
        return sorting_true,sorting_deep,recording_f,perf, comp

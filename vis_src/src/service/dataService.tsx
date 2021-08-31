import axios from 'axios';

// If enable, then it will load static data. Otherwise, it will load from remote backend.
// Currently, it does not support remote backend mode. 
const ENABLE_STATIC_JSON = false;
// Remote Backend.
const URL = process.env.NODE_ENV === 'development'
    ? 'http://localhost:7777'
    : window.location.origin;
const axiosInstance1 = axios.create({
    baseURL: `${URL}/api/`,
    // timeout: 1000,
    headers: {
        'Access-Control-Allow-Origin': '*'
    }
});

// Load Static Data.
const URL2 = window.location.origin;
const axiosInstance2 = axios.create({
    baseURL: `${URL2}/data/`,
    // timeout: 1000,
    headers: {
        'Access-Control-Allow-Origin': '*'
    }
});

let axiosInstance = (ENABLE_STATIC_JSON)?axiosInstance2:axiosInstance1;

// Read graph dataset metainfo.
export async function getDatasetList(): Promise<any> {
    let url = `/graphs`;
    if(ENABLE_STATIC_JSON){
        url = '/datasetlist.json'
    }
    //const params = { classifier_start, classifier_end };
    const res = await axiosInstance.get(url);
    if (res.status === 200) {
        return res.data;
    }
    throw res;
}

// Read model metainfo.
export async function getModelList(dataset_id:number): Promise<any> {
    let url = '/models'
    let res;
    if(ENABLE_STATIC_JSON){
         url = '/cache_modellist_'+dataset_id+'.json'
         res = await axiosInstance.get(url);
    }else{
         let params = { dataset_id };
         res = await axiosInstance.get(url, {params});
    }
    
    if (res.status === 200) {
        return res.data;
    }
    throw res;
}

// Read subgraph metainfo.
export async function getSubgraphList(dataset_id:number): Promise<any> {
    let url = '/subgraphs'
    let res;
    if(ENABLE_STATIC_JSON){
        url = '/cache_subgraphlist_'+dataset_id+'.json';
        res = await axiosInstance.get(url);
    }else{
        let params = { dataset_id };
        res = await axiosInstance.get(url, {params});
    }
    if (res.status === 200) {
        return res.data;
    }
    throw res;
}

// Read graph data.
export async function getGraphInfo(dataset_id:number): Promise<any> {
    let url = '/graphinfo';
    let res;
    if(ENABLE_STATIC_JSON){
        url = '/cache_graph_'+dataset_id+".json";
        res = await axiosInstance.get(url);
    }else{
        let params = { dataset_id };
        res = await axiosInstance.get(url, {params});
    }
    if (res.status === 200) {
        return res.data;
    }
    throw res;
}

// Read model data.
export async function getModelInfo(dataset_id:number, model_id:number): Promise<any> {
    let url = '/modelinfo';
    let res;
    if(ENABLE_STATIC_JSON){
        url = '/cache_graph_'+dataset_id+"_model_"+model_id+".json";
        res = await axiosInstance.get(url);
    }else{
        let params = { dataset_id, model_id };
        res = await axiosInstance.get(url, {params});
    }
    if (res.status === 200) {
        return res.data;
    }
    throw res;
}

// Read subgraph data.
export async function getSubgraphInfo(dataset_id:number, subgraph_id:number): Promise<any> {
    let url = '/subgraphinfo';
    let res;
    if(ENABLE_STATIC_JSON){
        url = 'cache_graph_'+dataset_id+"_subgraph_"+subgraph_id+".json";
        res = await axiosInstance.get(url);
    }else{
        let params = { dataset_id, subgraph_id };
        res = await axiosInstance.get(url, {params});
    }
    if (res.status === 200) {
        return res.data;
    }
    throw res;
}

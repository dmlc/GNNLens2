// 定义 State 结构类型
export type StoreState = {
    // model : number | null,    
    dataset_id : number | null,
    // eweight_id : number | null,
    refreshnumber: number,
    datasetList : any[],
    NLabelList : any[],
    eweightList : any[],
    filters: any,
    selectedNodeIdList: any[],
    selectedMessagePassingNodeIdList: any[],
    specificNodeIdList: any[],

    select_inspect_node : number
    showSource: boolean,
    prevGraphJson: any,
    extendedMode:any,
    GraphViewSettingsModal_visible:boolean,
    GraphViewState:any,
    enableForceDirected:boolean,
    NLabelOptions:any,
    EWeightOptions:any
};
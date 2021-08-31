import { DATASET_ID_CHANGE, INIT_DATASET_LIST, 
    SELECTED_NODE_ID_LIST_CHANGE, 
    SHOW_SOURCE_CHANGE, SPECIFIC_NODE_ID_LIST_CHANGE, SELECT_INSPECT_NODE_CHANGE,
    CLEAR_ID_INFO,  PREV_GRAPH_JSON_CHANGE, EXTENDED_MODE_CHANGE, GRAPHVIEWSETTINGMODAL_VISIBLE_CHANGE,GRAPHVIEWSTATE_CHANGE, 
    NLABEL_CHANGE, EWEIGHT_CHANGE, ENABLE_FORCE_DIRECTED_CHANGE} from '../constants';

export const changeDataset = (dataset_id:number | null) =>({
    type: DATASET_ID_CHANGE,
    dataset_id: dataset_id
})

export const changeSelectedNodeIdList = (selectedNodeIdList: any) =>({
    type: SELECTED_NODE_ID_LIST_CHANGE,
    selectedNodeIdList: selectedNodeIdList
})
export const changeExtendedMode = (extendedMode: number) =>({
    type: EXTENDED_MODE_CHANGE,
    extendedMode: extendedMode
})
export const changeGraphViewSettingsModal_visible = (visible:boolean) =>({
    type: GRAPHVIEWSETTINGMODAL_VISIBLE_CHANGE,
    GraphViewSettingsModal_visible: visible
})
export const changeGraphViewState = (state_dict:any) =>({
    type: GRAPHVIEWSTATE_CHANGE,
    GraphViewState: state_dict
})
export const changeEnableForceDirected = (enableForceDirected:any) =>({
    type: ENABLE_FORCE_DIRECTED_CHANGE,
    enableForceDirected: enableForceDirected
})


export const changeSpecificNodeIdList = (specificNodeIdList: any) =>({
    type: SPECIFIC_NODE_ID_LIST_CHANGE,
    specificNodeIdList: specificNodeIdList
})

export const changeSelectInspectNode = (select_inspect_node:any)=>({
    type: SELECT_INSPECT_NODE_CHANGE,
    select_inspect_node: select_inspect_node
})

export const changeShowSource = (showSource: boolean) =>({
    type: SHOW_SOURCE_CHANGE,
    showSource: showSource
})
export const changePrevGraphJson = (prevGraphJson: any) =>({
    type: PREV_GRAPH_JSON_CHANGE,
    prevGraphJson: prevGraphJson
})

export const clearIdInfo = () =>({
    type: CLEAR_ID_INFO
})

export const initDatasetList = (datasetList:any) =>({
    type: INIT_DATASET_LIST,
    datasetList: datasetList
})

export const changeNLabel = (NLabelList: any) =>({
    type: NLABEL_CHANGE,
    NLabelList: NLabelList
})

export const changeEWeight = (eweightList: any) =>({
    type: EWEIGHT_CHANGE,
    eweightList: eweightList
})


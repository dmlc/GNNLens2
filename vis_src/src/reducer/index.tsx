import { DATASET_ID_CHANGE, INIT_DATASET_LIST,  SELECTED_NODE_ID_LIST_CHANGE,
   SHOW_SOURCE_CHANGE,SPECIFIC_NODE_ID_LIST_CHANGE, SELECT_INSPECT_NODE_CHANGE,CLEAR_ID_INFO, PREV_GRAPH_JSON_CHANGE,
 GRAPHVIEWSTATE_CHANGE, GRAPHVIEWSETTINGMODAL_VISIBLE_CHANGE, EXTENDED_MODE_CHANGE, 
 NLABEL_CHANGE, EWEIGHT_CHANGE, ENABLE_FORCE_DIRECTED_CHANGE, NLABEL_OPTIONS_CHANGE, EWEIGHT_OPTIONS_CHANGE} from '../constants';
import {StoreState} from '../types';
const initial_state : StoreState = {
    dataset_id : null,
    refreshnumber: 0,
    showSource: false,
    datasetList: [],
    filters: {},
    selectedNodeIdList: [],
    selectedMessagePassingNodeIdList: [],
    specificNodeIdList: [],
    select_inspect_node : 0,
    prevGraphJson: null,
    extendedMode:1,
    GraphViewSettingsModal_visible:false,
    GraphViewState:{
      DisplayUnfocusedNodes:false,
      DisplayOverview:true
    },
    // model: null,
    NLabelList: [],
    // eweight_id: null,
    eweightList: [],
    enableForceDirected: true,
    NLabelOptions: [],
    EWeightOptions: []
}
// 处理并返回 state 
export default (state = initial_state, action:any): StoreState => {
   
    switch (action.type) {
      case DATASET_ID_CHANGE:
        // Change dataset id
        return {
          ...state,
          dataset_id: action.dataset_id,
        };
      case INIT_DATASET_LIST:
        // init dataset list
        return {
          ...state,
          datasetList: action.datasetList
        };
      case NLABEL_CHANGE:
        // Change NLabelList
        return {
          ...state,
          NLabelList: action.NLabelList
        }
      case EWEIGHT_CHANGE:
        // Change eweightList
        return {
          ...state,
          eweightList: action.eweightList
        }        
      case SELECTED_NODE_ID_LIST_CHANGE:
        //console.log("selectedNodeIdList Store State Change",action.selectedNodeIdList);
        return {
          ...state,
          selectedNodeIdList: action.selectedNodeIdList
        }
      case SHOW_SOURCE_CHANGE:
        return {
          ...state,
          showSource: action.showSource
        }
      case SPECIFIC_NODE_ID_LIST_CHANGE:
        //console.log("SpecificNodeIdListChange",  action.specificNodeIdList);
        return {
          ...state,
          specificNodeIdList: action.specificNodeIdList
        }
      case SELECT_INSPECT_NODE_CHANGE:
        //console.log("Select inspect node change", action.select_inspect_node);
        return {
          ...state,
          select_inspect_node: action.select_inspect_node
        }
      case CLEAR_ID_INFO:
        return {
          ...state,
          filters: {},
          selectedNodeIdList: [],
          selectedMessagePassingNodeIdList: [],
          specificNodeIdList: [],
          select_inspect_node : 0
        }
      case PREV_GRAPH_JSON_CHANGE:
        return {
          ...state,
          prevGraphJson: action.prevGraphJson
        }
      case GRAPHVIEWSETTINGMODAL_VISIBLE_CHANGE:
        return {
          ...state,
          GraphViewSettingsModal_visible: action.GraphViewSettingsModal_visible
        }
      case GRAPHVIEWSTATE_CHANGE:
        return {
          ...state,
          GraphViewState: action.GraphViewState
        }
      case EXTENDED_MODE_CHANGE:
          return {
            ...state,
            extendedMode: action.extendedMode
          }
      case ENABLE_FORCE_DIRECTED_CHANGE:
          return {
            ...state,
            enableForceDirected: action.enableForceDirected
          }
      case NLABEL_OPTIONS_CHANGE:
          return {
            ...state,
            NLabelOptions: action.NLabelOptions
          }
      case EWEIGHT_OPTIONS_CHANGE:
          return {
            ...state,
            EWeightOptions: action.EWeightOptions
          }
      default:
        return state
    }
}


//import { DECREMENT, INCREMENT } from '../constants';


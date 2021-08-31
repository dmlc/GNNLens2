import GraphView from '../components/DataRuns/GraphView/'
import { connect } from 'react-redux';
import { Dispatch } from 'redux';

//import { decrement, increment } from '../actions';
import { StoreState } from '../types';
import {changeSpecificNodeIdList, changeSelectInspectNode, changePrevGraphJson, 
    changeShowSource, changeExtendedMode, changeGraphViewSettingsModal_visible, changeEnableForceDirected} from '../actions';

// 将 reducer 中的状态插入到组件的 props 中
const mapStateToProps = (state: StoreState) => ({
    selectedNodeIdList : state.selectedNodeIdList,
    selectedMessagePassingNodeIdList: state.selectedMessagePassingNodeIdList,
    showSource: state.showSource,
    select_inspect_node: state.select_inspect_node,
    extendedMode : state.extendedMode,
    GraphViewSettingsModal_visible: state.GraphViewSettingsModal_visible,
    enableForceDirected : state.enableForceDirected
})

// 将 对应action 插入到组件的 props 中
const mapDispatchToProps = (dispatch: Dispatch) => ({
    changeSpecificNodeIdList:  (specificNodeIdList:any) => dispatch(changeSpecificNodeIdList(specificNodeIdList)),
    changeSelectInspectNode : (select_inspect_node:number) => dispatch(changeSelectInspectNode(select_inspect_node)),
    changePrevGraphJson: (prevGraphJson:any) => dispatch(changePrevGraphJson(prevGraphJson)),
    changeShowSource: (showSource:boolean) => dispatch(changeShowSource(showSource)),
    changeExtendedMode: (extendedMode:any) => dispatch(changeExtendedMode(extendedMode)),
    changeGraphViewSettingsModal_visible: (visible:boolean) => dispatch(changeGraphViewSettingsModal_visible(visible)),
    changeEnableForceDirected: (enableForceDirected: boolean) => dispatch(changeEnableForceDirected(enableForceDirected))
})

// 使用 connect 高阶组件对 Counter 进行包裹
export default connect(mapStateToProps, mapDispatchToProps)(GraphView);




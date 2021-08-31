import DataRuns from '../components/DataRuns'
import { connect } from 'react-redux';
import { Dispatch } from 'redux';

//import { decrement, increment } from '../actions';
import { StoreState } from '../types';
import {changeEnableForceDirected} from '../actions';

// 将 reducer 中的状态插入到组件的 props 中
const mapStateToProps = (state: StoreState) => ({
    dataset_id : state.dataset_id,
    NLabelList : state.NLabelList,
    eweightList: state.eweightList
})

// 将 对应action 插入到组件的 props 中
const mapDispatchToProps = (dispatch: Dispatch) => ({
    changeEnableForceDirected: (enableForceDirected: boolean) => dispatch(changeEnableForceDirected(enableForceDirected))

})

// 使用 connect 高阶组件对 Counter 进行包裹
export default connect(mapStateToProps, mapDispatchToProps)(DataRuns);




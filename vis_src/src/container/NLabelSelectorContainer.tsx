import NLabelSelector from '../components/DataRuns/ControlPanel/NLabelSelector';
import { connect } from 'react-redux';
import { Dispatch } from 'redux';

import {changeNLabel} from '../actions';
import { StoreState } from '../types';


// 将 reducer 中的状态插入到组件的 props 中
const mapStateToProps = (state: StoreState) => ({
    NLabelOptions : state.NLabelOptions,
    NLabelList : state.NLabelList,
})

// 将 对应action 插入到组件的 props 中
const mapDispatchToProps = (dispatch: Dispatch) => ({
    changeNLabel: (NLabelList: any | null) => dispatch(changeNLabel(NLabelList)),
})

// 使用 connect 高阶组件对 Counter 进行包裹
export default connect(mapStateToProps, mapDispatchToProps)(NLabelSelector);
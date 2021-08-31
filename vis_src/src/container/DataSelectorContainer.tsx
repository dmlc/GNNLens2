import DataSelector from '../components/DataRuns/ControlPanel/DataSelector';
import { connect } from 'react-redux';
import { Dispatch } from 'redux';

import {changeDataset,initDatasetList, clearIdInfo} from '../actions';
import { StoreState } from '../types';


// 将 reducer 中的状态插入到组件的 props 中
const mapStateToProps = (state: StoreState) => ({
    dataset_id: state.dataset_id,
    datasetList: state.datasetList
})

// 将 对应action 插入到组件的 props 中
const mapDispatchToProps = (dispatch: Dispatch) => ({
    changeDataset: (dataset_id:number | null) => dispatch(changeDataset(dataset_id)),
    clearIdInfo: () => dispatch(clearIdInfo()),
    initDatasetList : (datasetList: any) => dispatch(initDatasetList(datasetList))
})

// 使用 connect 高阶组件对 Counter 进行包裹
export default connect(mapStateToProps, mapDispatchToProps)(DataSelector);

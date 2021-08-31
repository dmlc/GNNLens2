import { connect } from 'react-redux';
import { Dispatch } from 'redux';
import { StoreState } from '../types';
import GraphViewSettingsModal from '../components/DataRuns/GraphView/GraphViewSettingsModal';
import { changeGraphViewSettingsModal_visible,changeGraphViewState } from '../actions';

const mapStateToProps = (state: StoreState) => ({
    GraphViewSettingsModal_visible: state.GraphViewSettingsModal_visible,
    GraphViewState: state.GraphViewState
})

const mapDispatchToProps = (dispatch: Dispatch) => ({
    changeGraphViewSettingsModal_visible: (visible:boolean) => dispatch(changeGraphViewSettingsModal_visible(visible)),
    changeGraphViewState: (state_dict:any) => dispatch(changeGraphViewState(state_dict))
})

export default connect(mapStateToProps, mapDispatchToProps)(GraphViewSettingsModal);




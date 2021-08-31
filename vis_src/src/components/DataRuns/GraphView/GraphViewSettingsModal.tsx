import * as React from "react";
import { Modal, Select, Button, Checkbox } from 'antd';
export interface IProps {
    GraphViewSettingsModal_visible: any,
    changeGraphViewSettingsModal_visible:any,
    GraphViewState:any,
    changeGraphViewState:any
}
export interface IState {
}
export default class GraphViewSettingsModal extends React.Component<IProps, IState>{
    constructor(props:IProps) {
        super(props);

        this.state = {
        }
        //this.resize.bind(this);
        // Flow:
        // 1. Constructor
        // 2. componentWillMount()
        // 3. render()
        // 4. componentDidMount()
        // If props update:
        // 4.1 componentWillReceiveProps(nextProps : IProps), then goto 5.
        // If States update
        // 5. shouldComponentUpdate() if return false, then no rerendering.
        // 6. if True, then componentWillUpdate
        // 7. render()
        // 8. componentDidUpdate
        // If Unmount, then componentWillUnmount()
    }
    handleOk = (e:any) => {
        console.log(e);
        this.props.changeGraphViewSettingsModal_visible(false);
      };
    
    handleCancel = (e:any) => {
        console.log(e);
        this.props.changeGraphViewSettingsModal_visible(false);
      };
    handleUnfocusedNodesChange = (e:any) =>{
      let checked = e.target.checked;
      let GraphViewState = this.props.GraphViewState;
      this.props.changeGraphViewState({
        ...GraphViewState,
        DisplayUnfocusedNodes: checked
      })
    }
    handleOverviewChange = (e:any) =>{
      let checked = e.target.checked;
      let GraphViewState = this.props.GraphViewState;
      this.props.changeGraphViewState({
        ...GraphViewState,
        DisplayOverview: checked
      })
    }
    public render() {
        /**
         * 1. Show Overview of Graph?
               2. Show background?
               3. Show color legend?
               4. Show glyph legend?
               5. Extended?
               6. Max Node Settings?
         */
        let GraphViewState = this.props.GraphViewState;
        return  (      
        <Modal
            title="Graph View Settings"
            visible={this.props.GraphViewSettingsModal_visible}
            onOk={this.handleOk}
            onCancel={this.handleCancel}
            footer={[
                <Button key="OK" type="primary" onClick={this.handleOk}>
                  OK
                </Button>
              ]}
        >
          
              Rendering Options:
              <div>
                <Checkbox checked={GraphViewState.DisplayUnfocusedNodes} onChange={this.handleUnfocusedNodesChange}>Display unfocused nodes.</Checkbox>
                <br />
                <Checkbox checked={GraphViewState.DisplayOverview} onChange={this.handleOverviewChange}>Display overview.</Checkbox>
              </div>
              
               
               
        </Modal>)
    }
}


import * as React from 'react';
import {Row} from 'antd';
import DataSelectorContainer from '../../../container/DataSelectorContainer';
import NLabelSelectorContainer from '../../../container/NLabelSelectorContainer';
import EWeightSelectorContainer from '../../../container/EWeightSelectorContainer';
export interface ControlPanelProps {
    nlabel_options:any[],
    eweight_options:any[]
}

export interface ControlPanelState {}

export default class ControlPanel extends React.Component<ControlPanelProps, ControlPanelState> {
    constructor(props: ControlPanelProps) {
        super(props);
        this.state = {
            
        };
    }
    public render() {
        let {nlabel_options, eweight_options} = this.props;

        // Generate NLabelSelector.
        let generateNLabelSelector = (nlabel_options: any) => {
            return <NLabelSelectorContainer nlabel_options={nlabel_options}/>
        }

        // Generate EWeightSelector.
        let generateEWeightSelector = (eweight_options: any) => {
            return <EWeightSelectorContainer eweight_options={eweight_options}/>
        }

        return (
            <div>
            <div className="ViewTitle">Control Panel</div>
            <div className="ViewBox">
                    <Row>
                        <DataSelectorContainer />
                    </Row>
                    <Row>
                        {generateNLabelSelector(nlabel_options)}
                    </Row>
                    <Row>
                        {generateEWeightSelector(eweight_options)}
                    </Row>
            </div>
            </div>
            
        );
    }
}

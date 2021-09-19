import * as React from 'react';
import {Row} from 'antd';
import DataSelectorContainer from '../../../container/DataSelectorContainer';
import NLabelSelectorContainer from '../../../container/NLabelSelectorContainer';
import EWeightSelectorContainer from '../../../container/EWeightSelectorContainer';
export interface ControlPanelProps {
    dataset_id: number | null,
}

export interface ControlPanelState {}

export default class ControlPanel extends React.Component<ControlPanelProps, ControlPanelState> {
    constructor(props: ControlPanelProps) {
        super(props);
        this.state = {};
    }
    public render() {
        return (
            <div>
            <div className="ViewTitle">Control Panel</div>
            <div className="ViewBox">
                    <Row>
                        <DataSelectorContainer />
                    </Row>
                    <Row>
                        <NLabelSelectorContainer />
                    </Row>
                    <Row>
                        <EWeightSelectorContainer />
                    </Row>
            </div>
            </div>
            
        );
    }
}

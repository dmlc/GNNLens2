import * as React from 'react';
import { Select, Row } from 'antd';
const Option = Select.Option;

export interface EWeightSelectorProps {
    eweight_options: any[], // eweight options
    changeEWeight: any,  // change eweight
}

export interface EWeightSelectorState {
}

export default class EWeightSelector extends React.Component<EWeightSelectorProps, EWeightSelectorState> {
    constructor(props: EWeightSelectorProps) {
        super(props);
        this.onEWeightSelectorChange = this.onEWeightSelectorChange.bind(this);
        this.state = {
        };
    }
        
    public onEWeightSelectorChange(value: any[]) {
        this.props.changeEWeight(value);
    }
    public render() {        
        let {eweight_options} = this.props;
        let disabledEWeightSelector = eweight_options.length <= 0
        let eweight_options_indexed = [];
        for(let i = 0; i<eweight_options.length; i++){
            let eweight_object:any = {
                "name": eweight_options[i],
                "id": i
            }
            eweight_options_indexed.push(eweight_object);
        }
        return (
                <Row>
                        EWeight:&nbsp;
                        <Select
                            allowClear
                            placeholder="Select eweights"
                            style={{ width: '170px' }}
                            onChange={this.onEWeightSelectorChange}
                            disabled={disabledEWeightSelector}
                        >
                            {eweight_options_indexed.map((d:any)=>(
                                <Option value={d.name} key={d.id}>{d.name}</Option>
                            ))}
                        </Select>
                </Row>
                
            )
            
    }
}
import * as React from 'react';
import { Select, Row } from 'antd';
const Option = Select.Option;

export interface IProps {
    EWeightOptions: any[], // eweight options
    eweightList: any,      // selected eweight options
    changeEWeight: any,    // change eweight
}

export interface IState {}

export default class EWeightSelector extends React.Component<IProps, IState> {
    constructor(props: IProps) {
        super(props);
        this.onEWeightSelectorChange = this.onEWeightSelectorChange.bind(this);
        this.state = {
        };
    }
        
    public onEWeightSelectorChange(value: any[]) {
        this.props.changeEWeight(value);
    }
    public render() {        
        let {EWeightOptions} = this.props;
        let disabledEWeightSelector = EWeightOptions.length <= 0
        let eweight_options_indexed = [];
        for(let i = 0; i<EWeightOptions.length; i++){
            let eweight_object:any = {
                "name": EWeightOptions[i],
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
                            value={this.props.eweightList}
                        >
                            {eweight_options_indexed.map((d:any)=>(
                                <Option value={d.name} key={d.id}>{d.name}</Option>
                            ))}
                        </Select>
                </Row>
                
            )
            
    }
}
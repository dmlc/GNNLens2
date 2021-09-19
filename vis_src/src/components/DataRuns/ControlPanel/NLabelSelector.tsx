import * as React from 'react';
import { Select, Row } from 'antd';
const Option = Select.Option;

export interface IProps {
    NLabelOptions: any[], // nlabel options
    NLabelList : any,     // selected nlabel options
    changeNLabel: any   // change nlabel
}

export interface IState {}

export default class NLabelSelector extends React.Component<IProps, IState> {
    constructor(props: IProps) {
        super(props);
        this.onNLabelSelectorChange = this.onNLabelSelectorChange.bind(this);
        this.state = {};
    }
    public onNLabelSelectorChange(value: any[]) {
        this.props.changeNLabel(value);
    }
    public render() {
        let {NLabelOptions} = this.props;
        let disabledNLabelSelector =  NLabelOptions.length <= 0;
        let nlabel_options_indexed = [];
        for(let i = 0; i< NLabelOptions.length; i++){
            let nlabel_object:any = {
                "name":  NLabelOptions[i],
                "id": i
            }
            nlabel_options_indexed.push(nlabel_object);
        }
        return (
                <Row>
                    NLabel:&nbsp;
                    <Select
                        mode="multiple"
                        allowClear
                        placeholder="Select nlabels"
                        style={{ width: '170px' }}
                        onChange={this.onNLabelSelectorChange}
                        disabled={disabledNLabelSelector}
                        value={this.props.NLabelList}
                        defaultValue={[]}
                    >
                        {nlabel_options_indexed.map((d:any)=>(
                            <Option value={d.name} key={d.id}>{d.name}</Option>
                        ))}
                    </Select>
                </Row>                
            )
    }
}

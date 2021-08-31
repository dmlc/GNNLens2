import * as React from 'react';
import { Select, Row } from 'antd';
const Option = Select.Option;

export interface IProps {
    nlabel_options: any[], // nlabel options
    changeNLabel: any   // change nlabel
}

export interface IState {
}

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
        let {nlabel_options} = this.props;
        let disabledNLabelSelector = nlabel_options.length <= 0;
        let nlabel_options_indexed = [];
        for(let i = 0; i<nlabel_options.length; i++){
            let nlabel_object:any = {
                "name": nlabel_options[i],
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

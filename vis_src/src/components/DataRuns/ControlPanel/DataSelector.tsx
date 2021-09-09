import * as React from 'react';
import { Select, Row } from 'antd';
import { getDatasetList } from '../../../service/dataService';
const Option = Select.Option;

export interface DataSelectorProps {
    dataset_id : number | null,  // dataset id
    datasetList: any,   // dataset list
    changeDataset: any,  // change dataset
    clearIdInfo: any,    // clear id info
    initDatasetList: any,  // initialize dataset list
}

export interface DataSelectorState {
}

export default class DataSelector extends React.Component<DataSelectorProps, DataSelectorState> {
    constructor(props: DataSelectorProps) {
        super(props);
        this.onDatasetSelectorChange = this.onDatasetSelectorChange.bind(this);
        this.state = {
        };
    }
    // Initialize Dataset List.
    componentDidMount(){
        this.initDatasetList();
    }
    public async initDatasetList(){
        const datasetList_package = await getDatasetList();
        if(datasetList_package["success"] === true){
            this.props.initDatasetList(datasetList_package["datasets"]);
        }
        
    }
    // Handling the event of changing data selector. 
    public onDatasetSelectorChange(value: number) {
        this.props.changeDataset(value);
        this.props.clearIdInfo();
    }
    public render() {        
        let disabledDatasetSelector = this.props.datasetList.length <= 0;
        return (
                <Row>
                        Graph:&nbsp;
                        <Select
                            placeholder="Select a graph"
                            value={this.props.dataset_id  || undefined}
                            style={{ width: '170px' }}
                            onChange={this.onDatasetSelectorChange}
                            disabled={disabledDatasetSelector}
                        >
                            {this.props.datasetList.map((d:any)=>(
                                <Option value={d.id} key={d.id}>
                                    {d.name}
                                </Option>
                            ))}
                        </Select>
                </Row>
                
            )
            
    }
}
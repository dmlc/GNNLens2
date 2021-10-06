
import * as React from "react";
import { Select, Button,  Tag, InputNumber } from 'antd';
import {getNodeColor, constructNeighborSet, getNodeStatisticStr, 
    constructGraphIn, skew_weight} from '../../../helper';
import { SettingOutlined } from '@ant-design/icons';
import GraphViewSettingsModalContainer from '../../../container/GraphViewSettingsModalContainer';
import ForceDirectedGraphCanvasContainer from '../../../container/ForceDirectedGraphCanvasContainer';
const Option = Select.Option;


export interface IProps {
    // For original graph object
    graph_object:any,
    model_nlabels:any,
    model_eweights:any,
    model_nweights:any,
    NLabelList:any,
    eweightList:any,
    // For subgraphs
    subgs:any,
    subgList:any,
    // For size of view.
    width: number,
    height: number,
    // For displayed nodes.
    selectedNodeIdList:any[],
    // For users selected node.
    changeSelectInspectNode:any,
    select_inspect_node : number,
    changeShowSource:any,
    showSource: boolean,
    // For extended mode. 
    extendedMode: any,
    changeExtendedMode:any,
    // For Graph View Setting Modal.
    GraphViewSettingsModal_visible:any,
    changeGraphViewSettingsModal_visible:any,

    enableForceDirected: boolean,
    changeEnableForceDirected: any
}
export interface IState {

}

export default class GraphView extends React.Component<IProps, IState>{
    public prevGraphJson:any = null;
    constructor(props:IProps) {
        super(props);
        this.onEnableForceDirected = this.onEnableForceDirected.bind(this);
        this.onExtendedModeChange = this.onExtendedModeChange.bind(this);
        this.onNodeClick = this.onNodeClick.bind(this);
        this.onChangeSelectInspectNode = this.onChangeSelectInspectNode.bind(this);
        this.UpdateCurrentGraphJson = this.UpdateCurrentGraphJson.bind(this);
        this.state = {
        }
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
    public UpdateCurrentGraphJson(current_graph_json:any){
        //console.log("Store Graph Json.")
        this.prevGraphJson = current_graph_json;
    }
    // Handling the node click event.
    public onNodeClick(node_id:number){
        let {showSource} = this.props;
        if(showSource === false){
            // select node in graph view and update showSource mode.
            this.props.changeSelectInspectNode(node_id);
            this.props.changeShowSource(true);
            this.props.changeExtendedMode(3);
        }else{
            this.props.changeSelectInspectNode(node_id);
        } 
    }
    // Color Legend Info.
    public getColorLegend(color_mode:boolean, num_types:number){
        let label = [];
        if (color_mode) {
            for(let i = 0; i< num_types; i++){
                label.push({
                    "text":i,
                    "color":getNodeColor(i,2)
                })
            }
        }
        return label;
    }
    // Data Preprocessing for node subgraph
    //    graph_object: original graph object.
    //    model_nlabels: record mapping from model names to predicted nlabels
    //    model_eweights: record mapping from eweight names to eweight values
    //    model_nweights: record mapping from nweight names to nweight values
    //    NLabelList: array of nlabel names, e.g. "ground_truth", "GCN"
    //    eweightList: array of attention head names, e.g. "layer-0-head-1"
    //    selecteedNodeIdList: displayed nodes.
    //    enableForceDirected:  control whether using force directed layout.
    //    select_inspect_node:  users selected node id. 
    //    showSource:   current mode on whether showing only selected node. 
    //    width / height: the size of graph view.
    public constructNodeGraphJson(graph_object:any, model_nlabels:any, model_eweights:any, model_nweights:any, NLabelList:any, eweightList:any, selectedNodeIdList:any, 
        enableForceDirected:boolean, select_inspect_node:number, showSource:boolean, width:number, height:number){
        let ew = eweightList;
        let selectStr = selectedNodeIdList.join("_");
        let NLabelName = NLabelList.join("_");
        let common = graph_object;
        
        // 1. Data package fingerprint. 
        let graph_name; 
        graph_name = common.name+"_"+common.bundle_id
                    +"_SELECTED_"+selectStr+"_SELECTEDEND_"
                    +"_NLABEL_"+NLabelName+"_NLABELEND_"
                    +"_EWEIGHT_"+ew+"_EWEIGHTEND_"
                    +enableForceDirected+"_"+width+"_"+height+"_";
        let graph_in = constructGraphIn(common);
        let graph_target = common.nlabels;
        // let graph_layout = common.layout;
        let source_list = graph_in.senders;
        let target_list = graph_in.receivers;
        let node_num = graph_in.num_nodes;
        let edge_num = graph_in.senders.length;
        let eweight = model_eweights;
        // let nweight = model_nweights;

        // 2. Default to show all nodes.
        if(selectedNodeIdList.length === 0){            
            selectedNodeIdList = []
            for(let i = 0; i<node_num;i++){
                selectedNodeIdList.push(i);
            }
        }

        // 3. Transform the graph layout.
        // let new_graph_layout = [];
        // for(let i = 0; i<node_num;i++){
        //    let xy = graph_layout[""+i];
        //    new_graph_layout.push(xy)
        // }
        // let enable_forceDirected = enableForceDirected;
        // if(new_graph_layout.length > 0){
        //    new_graph_layout = transform_graphlayout(new_graph_layout, width, height);
        // }

        // 4. If we have processed the graph layout, then we use previous graph layout.
        let enablePrevGraphLayout = false;
        let prevGraphJson = this.prevGraphJson;
        if(prevGraphJson && prevGraphJson["success"]){
            if(prevGraphJson["nodes"].length === node_num){
                enablePrevGraphLayout = true;
            }
        }
        //console.log("enablePrevGraphLayout, enableForceDirected, prevGraphJson", enablePrevGraphLayout, enableForceDirected, prevGraphJson);
        // 5. Derive the info of nodes and links.
        let nodes_json = [];   // node info
        let links_json = [];   // link info
        let links_color_json = [];   // link color info
        
        // Prepare properties of nodes.
        let color_mode: boolean = NLabelList.length !== 0;
        // If ground truth is not selected, then we use dummycolor to fill the area of ground truth. 
        let dummycolor = "#aaa";
        let init_color:any = [];
        if (!color_mode) {
            init_color = [dummycolor];
        }
        for(let i = 0; i<node_num;i++){
            let label = 0;
            let index = i;
            let real_color:any;
            let highlight = 1;
            let node_weight = 1;
            let color = init_color.slice();
            NLabelList.forEach((d:any)=>{
                if(d === "ground_truth"){
                    let nlabel = graph_target[index];
                    color.push(getNodeColor(nlabel, 2));
                }else{
                    let nlabel = model_nlabels[d][index];
                    color.push(getNodeColor(nlabel, 2));
                }
            })
            real_color = color.slice();  // original color storage.
            if(selectedNodeIdList.indexOf(index)<0){
                // Unfocused nodes color will be set to "#ddd".
                color = ["#ddd"];
                highlight = 0;
            }
            
            //if(nweight && NLabelList[0] && nweight[NLabelList[0]]) {
            //    node_weight = nweight[NLabelList[0]][index];
            //}
            let radius = 3;
            if(index === select_inspect_node && showSource === true){
                radius = 6;
            }
            let node_object:any = {
                "id":index,
                "group":label, // dummy
                "color":color,
                "real_color":real_color,
                "radius":radius,  // the radius of the node
                "highlight":highlight,  // whether the node is highlighted.
                "node_weight":skew_weight(node_weight)
            }            
            if(enablePrevGraphLayout){
                node_object["x"] = prevGraphJson["nodes"][i]["x"];
                node_object["y"] = prevGraphJson["nodes"][i]["y"];
            }
            //else if(enable_forceDirected === false){
            //    node_object["x"] = new_graph_layout[i][0];
            //    node_object["y"] = new_graph_layout[i][1];
            //}
            nodes_json.push(node_object);
        }

        // Prepare Properties of Links
        let edge_weighted: boolean;
        let current_eweights;
        if(eweightList && eweightList.length!==0) {
            edge_weighted = true;
            let graph_eweight_options = Object.keys(common.eweights);
            if (graph_eweight_options.indexOf(eweightList) > -1) {
                current_eweights = common.eweights[eweightList];
            } else {
                current_eweights = eweight[eweightList];
            }
        } else {
            edge_weighted = false;
        }

        for(let i = 0; i<edge_num;i++){
            let link_color = "#eee";
            // TODO: make default width configurable
            let edge_weight = 0.1;
            if(edge_weighted) {
                edge_weight = current_eweights[i] * 0.6;
            }
            let real_color = "#bbb";
            if(selectedNodeIdList.indexOf(source_list[i])>=0){
                if(selectedNodeIdList.indexOf(target_list[i])>=0){
                    link_color = "#bbb";
                }
            }
            // Store the possible color. 
            if(links_color_json.indexOf(link_color)>=0){
                
            }else{
                links_color_json.push(link_color);
            }
            
            links_json.push({
                "source": source_list[i],
                "target": target_list[i],
                "value":1,
                "weight":skew_weight(edge_weight),
                "color":link_color,
                "real_color":real_color    // For hovered link color.
            })
        }

        let graph_json = {
            "success":true,
            "name":graph_name,
            "nodes":nodes_json,
            "links":links_json,
            "links_color":links_color_json,
            "nodenum":node_num,
            "edgenum":edge_num,
            "enable_forceDirected":enableForceDirected,
            "colorLegend":this.getColorLegend(color_mode, common.num_nlabel_types)
        }
        return graph_json;
    }

    // Data Preprocessing for edge subgraph
    //    graph_object: original graph object.
    //    model_nlabels: record mapping from model names to predicted nlabels
    //    NLabelList: array of nlabel names, e.g. "ground_truth", "GCN"
    //    select_inspect_node: selected node
    //    subg_name: subgraph name
    //    subgs: record mapping from subgraph name to subgraph collections
    //    enableForceDirected:  control whether using force directed layout.
    //    showSource:   current mode on whether showing only selected node.
    //    width / height: the size of graph view.
    public constructEdgeGraphJson(graph_object:any, model_nlabels:any, NLabelList:any, select_inspect_node:number, 
        subg_name:string, subgs:any, enableForceDirected:boolean, showSource:boolean, width:number, height:number){
        let NLabelName = NLabelList.join("_");
        let common = graph_object;

        // 1. Data package fingerprint. 
        let graph_name = common.name+"_"+common.bundle_id
                        +"_SUBG_"+subg_name+"_SUBGEND_"
                        +"_NLABEL_"+NLabelName+"_NLABELEND_"
                        +"_NODE_"+select_inspect_node+"_NODEEND_"
                        +enableForceDirected+"_"+width+"_"+height+"_";
        let node_num = common.num_nodes;
        let edge_num = common.srcs.length;
        let graph_target = common.nlabels;
        // let graph_layout = common.layout;
        let source_list = common.srcs;
        let target_list = common.dsts;
        // ordered
        let selectedNodeIdList = subgs[subg_name][select_inspect_node].nodes;
        // ordered
        let selectedEdgeIdList = subgs[subg_name][select_inspect_node].eids;
        let nweight = subgs[subg_name][select_inspect_node].nweight;
        let eweight = subgs[subg_name][select_inspect_node].eweight;
        
        // 2. Transform the graph layout.
        // TODO: revisit and see if this is really necessary.
        // let new_graph_layout = [];
        // for(let i = 0; i<node_num;i++){
        //     let xy = graph_layout[""+i];
        //     new_graph_layout.push(xy)
        // }
        let enable_forceDirected = enableForceDirected;
        // if(new_graph_layout.length > 0){
        //     new_graph_layout = transform_graphlayout(new_graph_layout, width, height);
        // }

        // 3. If we have processed the graph layout, then we use previous graph layout.
        let enablePrevGraphLayout = false;
        let prevGraphJson = this.prevGraphJson;
        if(prevGraphJson && prevGraphJson["success"]){
            if(prevGraphJson["nodes"].length === node_num){
                enablePrevGraphLayout = true;
            }
        }

        // 4. Derive the info of nodes and links.
        let nodes_json = [];   // node info
        let links_json = [];   // link info
        let links_color_json = [];   // link color info

        // Prepare properties of nodes.
        let color_mode: boolean = NLabelList.length !== 0;
        // If ground truth is not selected, then we use dummycolor to fill the area of ground truth.
        let dummycolor = "#aaa";
        let init_color:any = [];
        if (!color_mode) {
            init_color = [dummycolor];
        }
        let selectedNodeOrder = 0;
        console.log('select_inspect_node', select_inspect_node);
        console.log('showSource', showSource);
        console.log('selectedNodeIdList', selectedNodeIdList);
        for(let i = 0; i<node_num;i++){
            let label = 0;
            let index = i;
            let real_color:any;
            let highlight = 1;
            let node_weight = 1;
            let color = init_color.slice();
            NLabelList.forEach((d:any)=>{
                if(d === "ground_truth"){
                    let nlabel = graph_target[index];
                    color.push(getNodeColor(nlabel, 2));
                }else{
                    let nlabel = model_nlabels[d][index];
                    color.push(getNodeColor(nlabel, 2));
                }
            })
            real_color = color.slice();  // original color storage.

            if(selectedNodeIdList[selectedNodeOrder] === i){
                node_weight = nweight[selectedNodeOrder];
                selectedNodeOrder = selectedNodeOrder + 1;
            } else {
                // Unfocused nodes color will be set to "#ddd".
                color = ["#ddd"];
                highlight = 0;
            }

            let radius = 3;
            if(index === select_inspect_node && showSource === true){
                radius = 6;
            }
            let node_object:any = {
                "id":index,
                "group":label, // dummy
                "color":color,
                "real_color":real_color,
                "radius":radius,  // the radius of the node
                "highlight":highlight,  // whether the node is highlighted.
                "node_weight":skew_weight(node_weight)
            }
            if(enablePrevGraphLayout){
                node_object["x"] = prevGraphJson["nodes"][i]["x"];
                node_object["y"] = prevGraphJson["nodes"][i]["y"];
            }
            //else if(enable_forceDirected === false){
            //    node_object["x"] = new_graph_layout[i][0];
            //    node_object["y"] = new_graph_layout[i][1];
            // }
            nodes_json.push(node_object);
        }

        // Prepare Properties of Links
        let selectedEdgeOrder = 0;
        for(let i = 0; i<edge_num;i++){
            let link_color = "#eee";
            // TODO: make default width configurable
            let edge_weight = 0.1;
            let real_color = "#bbb";
            if(i === selectedEdgeIdList[selectedEdgeOrder]) {
                link_color = "#bbb";
                edge_weight = eweight[selectedEdgeOrder] * 0.6;
                selectedEdgeOrder = selectedEdgeOrder + 1;
            }
            // Store the possible color. 
            if(links_color_json.indexOf(link_color)>=0){
                
            }else{
                links_color_json.push(link_color);
            }

            links_json.push({
                "source": source_list[i],
                "target": target_list[i],
                "value":1,
                "weight":skew_weight(edge_weight),
                "color":link_color,
                "real_color":real_color    // For hovered link color.
            })
        }

        let graph_json = {
            "success":true,
            "name":graph_name,
            "nodes":nodes_json,
            "links":links_json,
            "links_color":links_color_json,
            "nodenum":node_num,
            "edgenum":edge_num,
            "enable_forceDirected":enable_forceDirected,
            "colorLegend":this.getColorLegend(color_mode, common.num_nlabel_types)
        }
        return graph_json;
    }

    // Enable Force Directed Layout.
    public onEnableForceDirected(checked:boolean){
        console.log("Change State,", checked);
        /*this.setState({
            enableForceDirected: checked
        })*/
        this.props.changeEnableForceDirected(checked);
    }

    // Extended Mode Change
    public onExtendedModeChange(e:any){
        this.props.changeExtendedMode(e);
    }

    // Construct Extended Selected Node Id List
    public constructExtendedSelectedNodeIdList(selectedNodeIdList:any, NeighborSet:any){
        if(selectedNodeIdList.length === 0){
            return [];
        }else{
            
            let new_selectedNodeIdList = selectedNodeIdList.slice();
            for(let i = 0 ; i<selectedNodeIdList.length; i++){
                let nodeId = selectedNodeIdList[i];
                new_selectedNodeIdList = new_selectedNodeIdList.concat(NeighborSet[nodeId])
            }

            new_selectedNodeIdList = Array.from(new Set(new_selectedNodeIdList));
            return new_selectedNodeIdList;
        }
        
    }

    // change select node.
    public onChangeSelectInspectNode(node_id:any, node_num:number){
        let new_node_id:number = parseInt(node_id);
        if(!new_node_id || new_node_id<0){
            new_node_id = 0;
        }
        if(new_node_id>=node_num){
            new_node_id = node_num - 1;
        }
        console.log("graphview, new_node_id", new_node_id);
        this.props.changeSelectInspectNode(new_node_id);
    }

    // show graph view setting modal.
    public showGraphViewSettingModal(){
        this.props.changeGraphViewSettingsModal_visible(true);
    }
    public render() {
        let {graph_object, model_nlabels, model_eweights, model_nweights, subgs, NLabelList, eweightList, subgList, 
            selectedNodeIdList, showSource, select_inspect_node, width, height, extendedMode} = this.props;

        let onNodeClick = this.onNodeClick;
        let UpdateCurrentGraphJson = this.UpdateCurrentGraphJson;
        let specificNodeIdList = selectedNodeIdList;

        let common = graph_object;
        let graph_in = constructGraphIn(common);

        // Construct Neighbor Set
        let NeighborSet = constructNeighborSet(graph_in);
        
        // Define Force Directed Graph Size.
        let ForceDirectedWidth = width - 10;
        let ForceDirectedHeight = height - 50;
        if(showSource){
            if(width < 800 && width > 650){
                ForceDirectedHeight = height - 50 - 23;
            }else if(width <= 650){
                ForceDirectedHeight = height - 50 - 47;
            }
        }else{
            if(width < 650 && width > 550){
                ForceDirectedHeight = height - 50 - 23;
            }else if(width <= 550){
                ForceDirectedHeight = height - 50 - 47;
            }
        }


        // Preprocess Data.
        let graph_json:any;
        if(extendedMode <= 3) {
            // According to the showSource to determine the displayed node. 
            if(showSource){
                specificNodeIdList = [select_inspect_node];
            }
            // Extended Mode Configuration
            if(extendedMode === 2){
                specificNodeIdList = this.constructExtendedSelectedNodeIdList(specificNodeIdList, NeighborSet);
            }else if(extendedMode === 3){
                specificNodeIdList = this.constructExtendedSelectedNodeIdList(specificNodeIdList, NeighborSet);
                specificNodeIdList = this.constructExtendedSelectedNodeIdList(specificNodeIdList, NeighborSet);
            }

            graph_json = this.constructNodeGraphJson(graph_object, model_nlabels, model_eweights, model_nweights, 
                NLabelList, eweightList, specificNodeIdList, this.props.enableForceDirected, 
                select_inspect_node, showSource, ForceDirectedWidth, ForceDirectedHeight);
        } else {
            let subg_name = subgList[extendedMode-4];
            specificNodeIdList = subgs[subg_name][select_inspect_node].nodes;

            graph_json = this.constructEdgeGraphJson(graph_object, model_nlabels, NLabelList, select_inspect_node, 
                subg_name, subgs, this.props.enableForceDirected, showSource, 
                ForceDirectedWidth, ForceDirectedHeight);
        }
        
        // Store NeighborSet.
        graph_json["NeighborSet"] = NeighborSet;
        
        if(graph_json["success"]){
            // Store Graph Json.
            //console.log("Store Graph Json.")
            //this.prevGraphJson = graph_json;

            // Get node num.
            let nodenum: number = graph_json["nodenum"];

            // Extended Options
            let extendedOptions = [
                [1,"None"],
                [2,"One Hop"],
                [3,"Two Hop"]];
            
            for (var subg_type_id = 0; subg_type_id < subgList.length; subg_type_id++) {
                extendedOptions.push([subg_type_id + 4, subgList[subg_type_id]]);
            }

            // Event Handler for Starting or Stoping Layout.
            let stopLayout = () =>{
                this.onEnableForceDirected(false);
            }
            let startLayout = () =>{
                this.onEnableForceDirected(true);
            }

            // If showSource is true, then it means that currently the user has selected a node in the graph view.

            return (            
            <div style={{width: "100%", height:""+(this.props.height - 10)+"px", overflowX: "hidden"}}>
                <div className="ViewTitle clearfix">Graph View
                    <div style={{float:'right'}}>
                    &nbsp;&nbsp;&nbsp;&nbsp;
                    {/** Input Id */}
                    {(showSource)?[<span key={"span"+1}>Id:</span>,
                    <InputNumber min={0} max={nodenum} size="small" value={select_inspect_node} onChange={(e:any)=> {this.onChangeSelectInspectNode(e,nodenum);}} />,
                    <span key={"span"+3}>&nbsp;</span>,
                    <Button size="small" onClick={()=>{this.props.changeShowSource(false);this.props.changeExtendedMode(1);}}>X</Button> ]:[<span key={"span"+2}></span>]}
                    {/** Setting Modal */}
                    <GraphViewSettingsModalContainer />
                    &nbsp;&nbsp;&nbsp;&nbsp;
                    {/** Extended Selector */}
                    Subgraph:&nbsp;
                    <Select
                        placeholder="Select an extended mode"
                        value={extendedMode}
                        style={{ width: '120px' }}
                        onChange={this.onExtendedModeChange}
                        disabled={!showSource}
                        size="small"
                    >
                        {extendedOptions.map((d:any)=>(
                            <Option value={d[0]} key={d[0]}>
                                {d[1]}
                            </Option>
                        ))}
                        </Select>
                    {/** Force Directed Layout Enabler */}
                    &nbsp;&nbsp;&nbsp;&nbsp;
                    {(this.props.enableForceDirected)?
                        <Button type="primary" size="small" onClick={stopLayout}>Stop Simulation</Button>:
                        <Button type="default" size="small" onClick={startLayout}>Start Simulation</Button>}
                    {/** Setting Modal Button */}
                    &nbsp;&nbsp;&nbsp;&nbsp;
                    <Button type="default" size="small" onClick={()=>{this.showGraphViewSettingModal()}} ><SettingOutlined /></Button>
                    {/** Node Num Info */}
                    &nbsp;&nbsp;&nbsp;&nbsp;
                    #Nodes: <Tag>{getNodeStatisticStr(specificNodeIdList.length, nodenum)} </Tag>
                    
                    </div>
                </div>
                {/** Force Directed Graph */}
                <div className="ViewBox">
                    <div
                    style={{
                        width: '100%',
                        }}
                    >
                    <ForceDirectedGraphCanvasContainer graph_json={graph_json} 
                    width={ForceDirectedWidth} height={ForceDirectedHeight} 
                    onNodeClick={onNodeClick} UpdateCurrentGraphJson={UpdateCurrentGraphJson}/>
                    </div>
                </div>
            </div>
            )}else{
                return <div />
            }
    }
}


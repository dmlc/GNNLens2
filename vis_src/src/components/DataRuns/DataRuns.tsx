
import './DataRuns.css'
import * as React from "react";
import GraphViewContainer from '../../container/GraphViewContainer';
import ControlPanelContainer from '../../container/ControlPanelContainer';
import { getGraphInfo, getModelList, getModelInfo, getSubgraphList, getSubgraphInfo } from '../../service/dataService';
import GridLayout from "react-grid-layout";


export interface IProps {
    dataset_id : number | null,
    contentWidth:number,
    contentHeight:number,
    NLabelList : any,
    eweightList: any,
    changeEnableForceDirected : any,
    changeNLabelOptions: any,
    changeEWeightOptions: any,
    changeNLabel: any,
    changeEWeight: any,
    changeShowSource: any,
}
export interface IState {
    graph_object : any,
    model_list : any,
    model_nlabels : any,
    model_eweights: any,
    model_nweights: any,
    subg_list : any,
    subgs : any,
    layout_config: any,
    screenWidth: number,
    screenHeight: number
}

export default class DataRuns extends React.Component<IProps, IState>{
    public GraphViewRef:any;
    public ControlPanelRef: any;
    constructor(props:IProps) {
        super(props);
        this.onResizeStop = this.onResizeStop.bind(this);
        this.getLayoutConfigWithName = this.getLayoutConfigWithName.bind(this);
        this.getCurrentLayoutConfig = this.getCurrentLayoutConfig.bind(this);
        this.GraphViewRef = React.createRef();
        this.ControlPanelRef = React.createRef();
        let m_to_eweights: Record<string, Array<number>> = {};
        this.state = {
            graph_object:{
                model : -1,
                graph : -1,  
            },
            model_list: null,
            model_nlabels: null,
            model_eweights: m_to_eweights,
            model_nweights: null,
            subg_list: [],
            subgs: null,
            layout_config: null,
            screenWidth : 0,
            screenHeight: 0
        }
        // show_mode_specification
        // 1 -> graph_input
        // 2 -> graph_target
        // 3 -> graph_output
        // 4 -> Explain_mode
        // Explained_node, default for 0.

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


     // When the view is mounted, it will be executed.
     componentDidMount(){
        //window.addEventListener('resize', this.onResize)
         this.setState({
             layout_config: this.getWholeLayoutConfig(),
             screenHeight: window.innerHeight,
             screenWidth: window.innerWidth
         })
     }
     // Get graph data. 
     public async getGraphBundledData(dataset_id:number){
         let data = await getGraphInfo(dataset_id);
         let m_to_eweights: Record<string, Array<number>> = {};
         if(data["success"] === true){
            data["graph_obj"]["bundle_id"] = dataset_id;
            for (var head in data["graph_obj"]["eweights"]) {
                let mname = "Graph/".concat(head);
                m_to_eweights[mname] = data["graph_obj"]["eweights"][head]
            }
            data["graph_obj"]["eweights"] = m_to_eweights
            this.setState({
                graph_object: data["graph_obj"]
            })
            this.props.changeEnableForceDirected(true);
            this.props.changeNLabelOptions([]);
            this.props.changeNLabelOptions([]);
            this.props.changeNLabel([]);
            this.props.changeEWeight([]);
            this.props.changeShowSource(false);
         }
     }
     // Get associated model list and model data.
     public async getModelData(dataset_id:number){
         let mlist = await getModelList(dataset_id);
         if(mlist["success"] === true){
            let m_to_nlabels: Record<string, Array<number>> = {};
            let m_to_eweights: Record<string, Array<number>> = {};
            // let m_to_nweights: Record<string, Array<number>> = {};
            for (var model_info of mlist["models"]) {
                let mdata = await getModelInfo(dataset_id, model_info["id"]);
                m_to_nlabels[mdata["model_obj"]["name"]] = mdata["model_obj"]["nlabels"];
                // m_to_eweights[mdata["model_obj"]["name"]] = mdata["model_obj"]["eweight"];
                // m_to_nweights[mdata["model_obj"]["name"]] = mdata["model_obj"]["nweight"];
                for (var head in mdata["model_obj"]["eweights"]) {
                    let mname = mdata["model_obj"]["name"].concat("/");
                    m_to_eweights[mname.concat(head)]= mdata["model_obj"]["eweights"][head];
                }
            }
            this.setState({
                model_list: mlist["models"],
                model_nlabels: m_to_nlabels,
                model_eweights: m_to_eweights, // TODO
                // model_nweights: m_to_nweights// TODO
            })
         }
     }
     // Get associated subgraph list and subgraph data.
     public async getSubgraphData(dataset_id:number){
         let slist = await getSubgraphList(dataset_id);
         if(slist["success"] === true){
             // Name to subgraphs
             let n_to_subgs: Record<string, any> = {};
             for (var subg_info of slist["subgraphs"]) {
                 let sdata = await getSubgraphInfo(dataset_id, subg_info["id"]);
                 n_to_subgs[sdata["name"]] = sdata["node_subgraphs"];
             }
             this.setState({
                 subg_list: slist["subgraphs"].map((d:any)=>(d["name"])),
                 subgs: n_to_subgs
             })
         }
     }
     // Get width and height from view name. 
     public getLayoutConfigWithName(name:string){
         let width = 0;
         let height = 0;
         if(name === "GraphView"){
             if(this.GraphViewRef){
                 width = this.GraphViewRef.current.offsetWidth;
                 height = this.GraphViewRef.current.offsetHeight;
             }
         }else if(name === "ControlPanel"){
             if(this.ControlPanelRef){
                 width = this.ControlPanelRef.current.offsetWidth;
                 height = this.ControlPanelRef.current.offsetHeight;
             }
         }

         return {
             "width":width,
             "height":height
         }
     }
     // Get the whole layout config. 
     public getWholeLayoutConfig(){
        let viewName = ["GraphView", "ControlPanel"]; 
        let layout_config:any = {};
        viewName.forEach((d:any)=>{
            layout_config[d] = this.getLayoutConfigWithName(d);
        })
        return layout_config;
     }

     // Get layout config from view name. 
     public getCurrentLayoutConfig(name:string){
         let layout_config = this.state.layout_config;
        if(layout_config){
            if(layout_config[name]){
                return layout_config[name];
            }else{
                return null;
            }
        }else{
            return null;
        }
     }

     // Handling the changing of states or props.
     componentDidUpdate(prevProps:IProps, prevState:IState) {
        //console.log('Component did update!')
        // If the dataset_id has been changed. 
        if(prevProps.dataset_id !== this.props.dataset_id){
            // If the id is valid.
            if( this.props.dataset_id  && this.props.dataset_id>=0){
                // Get the graph data. 
                this.getGraphBundledData(this.props.dataset_id);
                // Get the model data associated with the graph.
                this.getModelData(this.props.dataset_id);
                // Get the subgraph data associated with the graph.
                this.getSubgraphData(this.props.dataset_id);
            }else{
                // Set to a dummy case.
                this.setState({
                    graph_object:{
                        model : -1,
                        graph : -1,  
                    }
                })
            }
        }

        // If the window is resized, update the layout config. 
        if(prevProps.contentHeight!==this.props.contentHeight
            || prevProps.contentWidth !== this.props.contentWidth){
                this.setState({
                    layout_config: this.getWholeLayoutConfig()
                })
            }   
     }

    // RESERVED: handling the layout change.
    public onLayoutChange(layout:any){}
    // For react-grid-layout, when the resizing is fixed, the layout configuration should be updated.
    public onResizeStop(layout:any){
        console.log("onResizeStop", layout);
        console.log("Layout", this.getWholeLayoutConfig());
        this.setState({
            layout_config : this.getWholeLayoutConfig()
        })
        //var width = document.getElementById('a').offsetWidth;
    }
    public render() {
        // Rendering.
        let {graph_object, model_nlabels, model_eweights} = this.state;
        let dataset_id = -1;
        if(graph_object.bundle_id){
            dataset_id = graph_object.bundle_id;      
            let nlabel_options: any[] = [];
            console.log('graph_object', graph_object);
            if (graph_object.nlabels.length !== 0) {
                nlabel_options.push("ground_truth");
            }
            if (model_nlabels !== null) {
                nlabel_options = nlabel_options.concat(Object.keys(model_nlabels));
            }
            this.props.changeNLabelOptions(nlabel_options);

            let eweight_options: any[] = [];
            eweight_options = Object.keys(graph_object.eweights);
            if (model_eweights !== null) {
                eweight_options = eweight_options.concat(Object.keys(model_eweights));
            }
            this.props.changeEWeightOptions(eweight_options);
        }

        // Generate Graph View.
        let generateGraphView = (graph_object: any, model_nlabels: any, model_eweights: any, model_nweights: any, 
            subgs: any, NLabelList: any, eweightList: any, subgList: any, width:number, height:number) => {
            return <GraphViewContainer graph_object={graph_object} 
                model_nlabels={model_nlabels}
                model_eweights={model_eweights}
                model_nweights={model_nweights}
                subgs={subgs}
                NLabelList={NLabelList}
                eweightList={eweightList}
                subgList={subgList}
                width={width}
                height={height}
                />
        }
        // Generate Control Panel
        let generateControlPanel = () => {
            return <ControlPanelContainer/>
        }
        
        // layout is an array of objects, see the demo for more complete usage
        let enableStatic = true;  // If enabled static, the layout cannot be manually configured.
        let max_row_num = Math.floor(this.props.contentHeight / 40); // Maximum rows in the screen.
        // small width, height: 1707 724
        // big width, height: 2560 1175
        let ControlPanelH = max_row_num;
        let GraphViewPanelH = max_row_num;
        
        let layout = [
            {i: 'b', x: 5, y: 0, w: 19, h: GraphViewPanelH, static:enableStatic}, // Graph View
            {i: 'd', x: 0, y: 0, w: 5, h: ControlPanelH, static:enableStatic}  // Control Panel
        ];
        

        // Generate Whole Layout.
        let generateWholeView = () =>{
            let screenwidth = window.innerWidth;
            //let screenheight = window.innerHeight;

            
            return <div><GridLayout className="layout" layout={layout} 
                cols={24} rowHeight={30} width={screenwidth} onLayoutChange={this.onLayoutChange}
                onResizeStop={this.onResizeStop}>
                    <div className="PanelBox" key="b" ref={this.GraphViewRef}>
                    {(dataset_id>=0 && this.getCurrentLayoutConfig("GraphView"))?generateGraphView(graph_object, 
                    this.state.model_nlabels, this.state.model_eweights, this.state.model_nweights, this.state.subgs, 
                    this.props.NLabelList, this.props.eweightList, this.state.subg_list,
                    this.getCurrentLayoutConfig("GraphView")["width"], 
                    this.getCurrentLayoutConfig("GraphView")["height"]):<div />}
                    </div>
                    <div className="PanelBox" key="d" ref={this.ControlPanelRef}>
                        {generateControlPanel()}
                    </div>
                </GridLayout>
                
                </div>
        }
        
        return generateWholeView();
    }
}


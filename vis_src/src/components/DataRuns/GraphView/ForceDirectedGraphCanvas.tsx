
import './ForceDirectedGraphCanvas.css'
import * as React from "react";
import {drawRectStroke, drawRect, drawNodeGlyph, drawLine} from './CanvasDrawing';
const d3 = require("d3");

export interface IProps {
    graph_json : any,
    width : number,
    height : number,
    onNodeClick: any,
    GraphViewState:any,
    UpdateCurrentGraphJson:any
}
export interface IState {
}

export default class ForceDirectedGraphCanvas extends React.Component<IProps, IState>{
    public global_simulation:any = null;
    public saved_transform:any = null;
    public refresh_number = 0;
    public current_graph_json:any = null;
    constructor(props:IProps) {
        super(props);
        this.updateTransform = this.updateTransform.bind(this);
        this.state = {

        }
    }
    componentDidMount(){
        this.renderCanvas();
    }
    
    componentDidUpdate(prevProps:IProps, prevState:IState) {
        if(prevProps.graph_json.name !== this.props.graph_json.name || prevProps.width !== this.props.width || prevProps.GraphViewState !== this.props.GraphViewState){
            this.renderCanvas();
        }
     }
     public updateTransform(transform:any){
         this.saved_transform = transform;
     }
     // Render Legend Function.
     public renderLegend(legend_configuration:any){
        var width = legend_configuration["width"];
        var height = legend_configuration["height"];
        var colorLegend = legend_configuration["colorLegend"];
        // ---------------- Color Legend -------------------------//
        let legend_pie_y = height - 10 - 100;
        var top_svg = d3.select("#force_directed_graph")
                .select("#svgChart")
                .attr("width", width)
                .attr("height", height);
        let legend_color_x = 10;
        let max_text_length = 0;
        colorLegend.forEach((d:any)=>{
            let text = "" + d.text;
            if(text.length>max_text_length){
                max_text_length = text.length;
            }
        })

        let legend_color_width = max_text_length*8+24;
        //console.log("maxtextlength", max_text_length, legend_color_width);
        let legend_color_height = colorLegend.length*20;
        let legend_color_y = legend_pie_y - legend_color_height - 10;
        var legend_color_svg = top_svg.select("#ForceDirectedColorLegend")
            .attr("width", legend_color_width)
            .attr("height", legend_color_height)
            .attr("transform", "translate("+legend_color_x+","+legend_color_y+")")
        let legend_rect = legend_color_svg.selectAll("rect").data([0]);
        let legend_rect_enter = legend_rect.enter().append("rect");
        //console.log("legend_rect", legend_rect);
        legend_rect_enter.merge(legend_rect)
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", legend_color_width)
            .attr("height", legend_color_height)
            .attr("fill", "#fff")
            .attr("opacity", 0.8)
            .attr("stroke", "#bbb")
            .attr("stroke-width", 1)
            .attr("rx",3)
            .attr("ry",3);
        let row_legend_color = legend_color_svg.selectAll("g.legend_row_color")
                                .data(colorLegend, function(d:any,i:any){
                                    return d.text+"_"+i+"_"+d.color;
                                });
        let g_row_legend_color = row_legend_color.enter().append("g")
                            .attr("class","legend_row_color")
                            .attr("transform", function(d:any,i:any){
                                return "translate(10,"+(10+i*20)+")";
                            });
            g_row_legend_color.append("circle")
                            .attr("r", 5)
                            .attr("fill", function(d:any){
                                return d.color;
                            })
                            
            g_row_legend_color.append("text")
                            .attr("x", 10)
                            .attr("y", 5)
                            .text(function(d:any){
                                return d.text;
                            })
                            
            row_legend_color.exit().remove();

        // ---------------- Edge Color Legend -------------------------//
        // let legend_edge_color_x = 10;
        // let legend_edge_color_width = 38;
        // let legend_edge_color_height = 100;
        // let legend_edge_color_y = legend_pie_y - legend_color_height - legend_edge_color_height - 20;
        // var legend_edge_color_svg = top_svg.select("#ForceDirectedEdgeColorLegend")
        //    .attr("width", legend_edge_color_width)
        //    .attr("height", legend_edge_color_height)
        //    .attr("transform", "translate("+legend_edge_color_x+","+legend_edge_color_y+")");
        // let legend_edge_rect = legend_edge_color_svg.selectAll("rect").data([0]);
        // let legend_edge_rect_enter = legend_edge_rect.enter().append("rect");
        // console.log("legend_rect", legend_rect);
        // legend_edge_rect_enter.merge(legend_edge_rect)
        //    .attr("x", 0)
        //    .attr("y", 0)
        //    .attr("width", legend_edge_color_width)
        //    .attr("height", legend_edge_color_height)
        //    .attr("fill", "#fff")
        //    .attr("opacity", 1)
        //    .attr("stroke", "#bbb")
        //    .attr("stroke-width", 1)
        //    .attr("rx",3)
        //    .attr("ry",3);
        // Create the svg:defs element and the main gradient definition.
        // var svgDefs = legend_edge_color_svg.selectAll("defs").data([0]);

        // var svgDefs_enter = svgDefs.enter().append('defs');

        // var mainGradient = svgDefs_enter.append('linearGradient')
        //    .attr('id', 'mainGradient')
        //    .attr("gradientTransform","rotate(90)");

        // Create the stops of the main gradient. Each stop will be assigned
        // a class to style the stop using CSS.
        // mainGradient.append('stop')
        //    .attr('stop-color', "rgba(187, 187, 187 ,1)")
        //    .attr('offset', '0');

        // mainGradient.append('stop')
        //    .attr('stop-color', 'rgba(187, 187, 187 ,0.1)')
        //    .attr('offset', '1');
        // let g_legend_edge_color = legend_edge_color_svg.selectAll("g.legend_edge_color").data([0]);
        // let g_legend_edge_color_enter = g_legend_edge_color.enter().append("g").attr("class","legend_edge_color");
        // let rect_legend_edge_color_enter = g_legend_edge_color_enter.append("rect");
        // let rect_legend_edge_color = g_legend_edge_color.select("rect");
        //let colorGradient = d3.scaleSequential(d3.interpolateRdYlGn);
        // let top_padding = 20;
        // let padding = 5;
        // let text_width = 10;
        // let rect_width = legend_edge_color_width-2*padding-text_width;
        // let rect_height = legend_edge_color_height-2*padding - top_padding;
        // rect_legend_edge_color_enter.merge(rect_legend_edge_color)
        //    .attr("x", padding)
        //    .attr("y", top_padding + padding)
        //    .attr("width", rect_width)
        //    .attr("height", rect_height)
        //    .attr("fill","url(#mainGradient)");
        // let legend_text_list = [
        //    {
        //        "text":"1",
        //        "x":padding + rect_width,
        //        "y":top_padding + padding,
        //        "dx":".02em",
        //        "dy":".65em",
        //        "text-anchor":"start"
        //    },
        //    {
        //        "text":"0",
        //        "x":padding + rect_width,
        //        "y":top_padding + padding + rect_height,
        //        "dx":".02em",
        //        "dy":".05em",
        //        "text-anchor":"start"
        //    },
        //    {
        //        "text":"Edge",
        //        "x":padding,
        //        "y":padding,
        //        "dx":".00em",
        //        "dy":".65em",
        //        "text-anchor":"start"
        //  }];
        // let legend_text_update = g_legend_edge_color_enter.selectAll("text").data(legend_text_list, function(d:any){
        //    return d.text;
        // });
        // let legend_text_enter = legend_text_update.enter().append("text");
        
        // legend_text_update.merge(legend_text_enter)
        //    .attr("x", (d:any)=>d.x)
        //    .attr("y", (d:any)=>d.y)
        //    .attr("dx", (d:any)=>d.dx)
        //    .attr("dy", (d:any)=>d.dy)
        //    .attr("text-anchor",(d:any)=>d["text-anchor"])
        //    .text((d:any)=>{
        //        return d.text;
        //    });
        // let legend_title = g_legend_edge_color.exit().remove(); 
     }
     // Render Canvas Main Function
     public renderCanvas(){

        // initialize 
        this.props.UpdateCurrentGraphJson(this.props.graph_json);
        var onNodeClick = this.props.onNodeClick;
        var nodenum = this.props.graph_json.nodenum;
        var enabledForceDirected = this.props.graph_json.enable_forceDirected;
        var neighborSet = this.props.graph_json.NeighborSet;
        var colorLegend = this.props.graph_json.colorLegend;
        var configuration = {
            "strength": 0.01,
            "radius":15,
            "showlabel": true,
            "showarrow": true,
            "width": this.props.width,
            "height": this.props.height
        }
        var GraphViewState = this.props.GraphViewState;
        var DisplayUnfocusedNodes = GraphViewState.DisplayUnfocusedNodes;
        var DisplayOverview = GraphViewState.DisplayOverview;
        //console.log("ForceDirected" , nodenum)
        if(nodenum >= 100){
            configuration = {
                "strength": 0.4,
                "radius":3,
                "showlabel": false,
                "showarrow": false,
                "width": this.props.width,
                "height": this.props.height
            }
        }

        var width = configuration["width"];
        var height = configuration["height"];
        var radius = configuration["radius"];
        var radius_gap = 0.3;
        var graphWidth =  this.props.width;
        var legend_configuration:any = {
            "width":width,
            "height":height,
            "colorLegend":colorLegend,
        }
        // Render Legend
        this.renderLegend(legend_configuration);

        // Layered Canvas
        //     Detect Event -- Capture Event
        //     Overview Canvas  -- Display Overview
        //     Hovered Canvas -- Hovered Canvas (will not modify the bottom canvas when hovering)
        //     Bottom Canvas -- Bottom Canvas


        // 1. Original Canvas
        var graphCanvas = d3.select('#force_directed_graph').select('#bottom')
        .attr('width', graphWidth + 'px')
        .attr('height', height + 'px')
        .node();
        
        var context = graphCanvas.getContext('2d');

        // 2. Hovered Canvas
        var middleCanvas = d3.select('#force_directed_graph').select("#middle")
        .attr('width', graphWidth + 'px')
        .attr('height', height + 'px')
        .node();
        var middle_context = middleCanvas.getContext('2d');

        // 3. Overview Canvas
        var overviewCanvas = d3.select('#force_directed_graph').select('#overview')
        .attr('width', graphWidth + 'px')
        .attr('height', height + 'px')
        .node();
        var overview_context = overviewCanvas.getContext('2d');

        // 4. Detect Event 
        var eventCanvas = d3.select('#force_directed_graph').select("#event")
        .attr('width', graphWidth + 'px')
        .attr('height', height + 'px')
        .node();
         /**
         * OverviewCanvas
         */
        let canvasWidth = 100;
        let canvasHeight = 100;
        let margin = 10;
        let canvasX = graphWidth - canvasWidth - margin;
        let canvasY = height - canvasHeight - margin;
        let canvasXRight = canvasX + canvasWidth;
        let canvasYBottom = canvasY + canvasHeight;
        let radius_collision = radius*3 + radius_gap*3;

        // Force Directed Layout Algorithm.
        if(this.global_simulation){
            this.global_simulation.stop();
            delete this.global_simulation;
        }
        var simulation = d3.forceSimulation()
                      .force("center", d3.forceCenter(graphWidth / 2, height / 2))
                      .force("x", d3.forceX(graphWidth / 2).strength(0.1))
                      .force("y", d3.forceY(height / 2).strength(0.1))
                      .force("charge", d3.forceManyBody().strength(-50))
                      .force("link", d3.forceLink().strength(1).id(function(d:any) { return d.id; }))
                      .force('collide', d3.forceCollide().radius((d:any) => radius_collision))
                      .alphaTarget(0)
                      .alphaDecay(0.05)
                      
        this.global_simulation = simulation;

        let updateTransform = this.updateTransform;

        // Transform
        //    This transform is preserved. 
        var transform:any;
        var calTransform:any={
            "x":0,
            "y":0,
            "k":1
        };
        if(this.saved_transform){
            transform =this.saved_transform ;
        }else{
            transform = d3.zoomIdentity;
        }

        
        // Judge the hovered flag.
        function judgeHoveredFlag(d:any, bool:boolean){
            if(!d.hasOwnProperty("hovered") || d["hovered"]===false ){
                if(bool === false){
                    return false;
                }else{
                    return true;
                }
            }else{
                if(bool === true){
                    return false;
                }else{
                    return true;
                }
            }
        }

        // Hide the tooltip. 
        function hiddenTooltip(){
            d3.select("#force_directed_graph").select("#tooltip").style('opacity', 0);

        }

        // Processed Data. 
        let tempData = this.props.graph_json;
        //console.log("tempData", tempData);

        let event_canvas = eventCanvas;
        var mouseCoordinates:any = null;
        d3.select(event_canvas).on("click",handleMouseClick).on("mousemove", handleMouseMove).on("mouseout",handleMouseOut);
        d3.select(event_canvas).call(d3.zoom().scaleExtent([1 / 10, 8]).on("zoom", zoomed))
        
        // Control whether using force directed layout. 
        // SimulationUpdate is used to render the layout. 
        if(enabledForceDirected){
            simulation
                .nodes(tempData.nodes)
                .on("tick", simulationUpdate);

            simulation.force("link")
                .links(tempData.links);

        }else{
            simulation.stop();
            simulation
                .nodes(tempData.nodes);

            simulation.force("link")
                .links(tempData.links);
            simulationUpdate();
        }
        
        

        // Determine the clicked ordering.
        function order_determine(a:any,b:any){
            let hover_cons_a = a.hasOwnProperty("hover_cons")?a.hover_cons:1;
            let hover_cons_b = b.hasOwnProperty("hover_cons")?b.hover_cons:1;
            let node_outer_radius_a = a.radius*hover_cons_a*2;
            let node_outer_radius_b = b.radius*hover_cons_b*2;
            return node_outer_radius_a<node_outer_radius_b?-1:1;
        }
        // Determine the clicked object.
        function determineSubject(mouse_x:number,mouse_y:number){
            var i,
            x = transform.invertX(mouse_x),
            y = transform.invertY(mouse_y),
            dx,
            dy;
            let newNodeList = tempData.nodes.slice().sort(order_determine)
            for (i = newNodeList.length - 1; i >= 0; --i) {
                var node = newNodeList[i];
                if(!DisplayUnfocusedNodes && !node["highlight"]){
                    continue;
                }
                dx = x - node.x;
                dy = y - node.y;
                let hover_cons = node.hasOwnProperty("hover_cons")?node.hover_cons:1;
                let outer_radius_node = node.radius * 2 * hover_cons;
                if (dx * dx + dy * dy < outer_radius_node * outer_radius_node) {
                    return node;
                }
            }
            return null;
        }
       


        // Zoom updating. 
        function zoomed(this:any) {
            var xy = d3.mouse(this);
            mouseCoordinates = xy;
            transform = d3.event.transform;
            if(determineEventSubject(xy[0], xy[1])==="GraphCanvas"){
                updateTransform(transform);
                simulationUpdate();
            }
        }
        
        // Mouse Move Handler. 
        // Use middleCanvasSimulationUpdate() to update the hovered nodes. 
        function handleMouseMove(this:any, obj:any=null, defaultUpdateFlag:boolean=false){
            var xy:any;
            if(obj){
                xy = mouseCoordinates;
            }else{
                xy = d3.mouse(this);
                mouseCoordinates = xy;
            }
            
            var updateFlag = defaultUpdateFlag;

            if(xy){
                let event_subject = determineEventSubject(xy[0], xy[1]);
                var selected = determineSubject(xy[0],xy[1]);
                if(event_subject==="GraphCanvas"&&selected){
                    updateFlag = true;
                    let target_id = selected.id;

                    d3.select("#force_directed_graph").select('#tooltip')
                        .style('opacity', 0.8)
                        .style('top', (xy[1] + 5) + 'px')
                        .style('left', (xy[0] + 5) + 'px')
                        .html(target_id);

                    let neighbor_id = neighborSet[selected.id];
                    tempData.nodes.forEach((d:any)=>{
                        if(target_id === d.id){
                            d.hovered = true;
                            d.hover_cons = 3;
                        }else  if(neighbor_id.indexOf(d.id)>=0){
                            d.hovered = true;
                            d.hover_cons = 2;
                        }else{   
                            d.hovered = false;
                            d.hover_cons = 1;
                        }
                    })
                    
                }else{
                    tempData.nodes.forEach((d:any)=>{
                        updateFlag = updateFlag || judgeHoveredFlag(d, false);
                        d.hovered = false;
                        d.hover_cons = 1;
                    })
                    hiddenTooltip();
                }
            }else{
                tempData.nodes.forEach((d:any)=>{
                    updateFlag = updateFlag || judgeHoveredFlag(d, false);
                    d.hovered = false;
                    d.hover_cons = 1;
                })
                hiddenTooltip();
            }
            
            if(updateFlag){
                middleCanvasSimulationUpdate()
            }
            
        }

        // Mouse Out handler. 
        function handleMouseOut(this:any, obj:any=null, defaultUpdateFlag:boolean=false){
            var updateFlag = defaultUpdateFlag;
            mouseCoordinates = null;
            tempData.nodes.forEach((d:any)=>{
                updateFlag = updateFlag || judgeHoveredFlag(d, false);
                d.hovered = false;
                d.hover_cons = 1;
            })
            hiddenTooltip();
            if(updateFlag){
                middleCanvasSimulationUpdate()
            }
        }

        // Determine the clicked canvas. 
        function determineEventSubject(mouse_x:number, mouse_y:number){
            if(mouse_x >= canvasX && mouse_x <=canvasXRight 
                && mouse_y >= canvasY && mouse_y <=canvasYBottom && DisplayOverview){
                    return "OverviewCanvas";
                }else{
                    return "GraphCanvas";
                }
        }
        // Mouse Click Handler.
        function handleMouseClick(this:any, obj:any=null, defaultUpdateFlag:boolean=false){
            if (d3.event.defaultPrevented) return; // zoomed

            var xy:any;
            if(obj){
                xy = mouseCoordinates;
            }else{
                xy = d3.mouse(this);
                mouseCoordinates = xy;
            }

            if(xy){
                if(determineEventSubject(xy[0],xy[1])==="OverviewCanvas"){
                    moveFocalPoint(xy[0], xy[1]);
                }else{
                    var selected = determineSubject(xy[0],xy[1]);
                    if(selected){
                        onNodeClick(selected.id);
                    }
                }

            }else{

            }
            
        }

        // ---- The following code is reserved for overview canvas calculation. NOT IMPORTANT ---- //
        // Calculate the bounding box of graph.
        function calculateGraphBoundingBox(){
            //let canvasWidth = graphWidth;
            //let canvasHeight = height;
            let minx=0, miny=0, maxx=0, maxy=0;
            let flag = false;
            tempData.nodes.forEach(function(d:any){
                if(DisplayUnfocusedNodes || (!DisplayUnfocusedNodes && d.highlight)){
                    let x = d.x;
                    let y = d.y;
                    if(!flag){
                        minx = x;
                        miny = y;
                        maxx = x;
                        maxy = y;
                        flag = true;
                    }else{
                        if(minx > x){
                            minx = x;
                        }
                        if(maxx < x){
                            maxx = x;
                        }
                        if(miny > y){
                            miny = y;
                        }
                        if(maxy < y){
                            maxy = y;
                        }
                    }
                }
                
            })
            let glyph_outer_radius = 3*2;
            let margin = 14;
            let leftbound = minx - glyph_outer_radius - margin;
            let upperbound = miny - glyph_outer_radius - margin;
            
            let occupyWidth = maxx - minx + glyph_outer_radius*2 + margin*2;
            let occupyHeight = maxy - miny + glyph_outer_radius*2 + margin*2;
            return {
                "leftbound":leftbound,
                "upperbound":upperbound,
                "occupyWidth":occupyWidth,
                "occupyHeight":occupyHeight
            }
        }

        // Calculate the transformed rects.
        function rectTransform(rect_configuration:any, transform:any){
            let rect_x = rect_configuration["x"];
            let rect_y = rect_configuration["y"];
            let rect_width = rect_configuration["width"];
            let rect_height = rect_configuration["height"];
            let dx = transform.x;
            let dy = transform.y;
            let scale = transform.k;
            let x = (rect_x*scale + dx) ;
            let y = (rect_y*scale + dy) ;
            let width = (rect_width) * scale;
            let height = (rect_height) * scale;
            return {
                "x":x,
                "y":y,
                "width":width,
                "height":height
            }
        }

        // Calculate the inversed transformed rects.
        function rectInverseTransform(rect_configuration:any, transform:any){
            let rect_x = rect_configuration["x"];
            let rect_y = rect_configuration["y"];
            let rect_width = rect_configuration["width"];
            let rect_height = rect_configuration["height"];
            let dx = -transform.x;
            let dy = -transform.y;
            let scale = 1/transform.k;
            let x = (rect_x + dx) * scale;
            let y = (rect_y + dy) * scale;
            let width = (rect_width) * scale;
            let height = (rect_height) * scale;
            return {
                "x":x,
                "y":y,
                "width":width,
                "height":height
            }
        }

        // Calculate the inversed transformed of points.
        function pointInverseTransform(point_configuration:any, transform:any){
            let point_x = point_configuration["x"];
            let point_y = point_configuration["y"];
            let dx = -transform.x;
            let dy = -transform.y;
            let scale = 1/transform.k;
            let x = (point_x + dx) * scale;
            let y = (point_y + dy) * scale;
            return {
                "x":x,
                "y":y
            }
        }

        // Move the Overview Focused View.
        function moveFocalPoint(mouse_x:number, mouse_y:number){
            let ori_point = {
                "x":graphWidth / 2,
                "y":height / 2
            }
            let ori_inverse_point = pointInverseTransform(ori_point, transform);
            
            let overview_point = {
                "x": mouse_x,
                "y": mouse_y
            }
            let overview_inverse_point = pointInverseTransform(overview_point, calTransform);
            let new_x = -(overview_inverse_point["x"] - ori_inverse_point["x"])*transform.k + transform.x;
            let new_y = -(overview_inverse_point["y"] - ori_inverse_point["y"])*transform.k + transform.y;
            console.log({
                ori_point, ori_inverse_point, overview_point, overview_inverse_point, new_x, new_y
            })
            transform.x = new_x;
            transform.y = new_y;
            updateTransform(transform);
            simulationUpdate();
            
        }
        // Calculate the rect in the overview. 
        function rectInverseTransformAndClip(rect_configuration:any,transform:any, bounding_box:any){
            let leftbound = bounding_box["leftbound"];
            let upperbound = bounding_box["upperbound"];
            let occupyHeight = bounding_box["occupyHeight"];
            let occupyWidth = bounding_box["occupyWidth"];
            let rightbound = leftbound + occupyWidth;
            let lowerbound = upperbound + occupyHeight;
            let inverse_transform_rect = rectInverseTransform(rect_configuration, transform);
            let transformed_leftbound = inverse_transform_rect["x"];
            let transformed_upperbound = inverse_transform_rect["y"];
            let transformed_rightbound = inverse_transform_rect["x"]+inverse_transform_rect["width"];
            let transformed_lowerbound = inverse_transform_rect["y"]+inverse_transform_rect["height"];

            if(transformed_leftbound<leftbound){
                transformed_leftbound = leftbound;
            }
            if(transformed_rightbound>rightbound){
                transformed_rightbound = rightbound;
            }
            if(transformed_upperbound<upperbound){
                transformed_upperbound = upperbound;
            }
            if(transformed_lowerbound>lowerbound){
                transformed_lowerbound = lowerbound;
            }
            let clipx = transformed_leftbound;
            let clipy = transformed_upperbound;
            let clipwidth = transformed_rightbound - transformed_leftbound;
            let clipheight = transformed_lowerbound - transformed_upperbound;
            if(clipwidth < 0){
                clipwidth = 0;
            }else if(clipwidth>occupyWidth){
                clipwidth = occupyWidth;
            }
            if(clipheight<0){
                clipheight=0;
            }else if(clipheight>occupyHeight){
                clipheight = occupyHeight;
            }
            return {
                "x":clipx,
                "y":clipy,
                "width":clipwidth,
                "height":clipheight
            }

        }
        function calculateTransform(canvasX:number,canvasY:number,canvasWidth:number, canvasHeight:number, bounding_box:any){
            let leftbound = bounding_box["leftbound"];
            let upperbound = bounding_box["upperbound"];
            let occupyHeight = bounding_box["occupyHeight"];
            let occupyWidth = bounding_box["occupyWidth"];
            let xscale = canvasWidth / occupyWidth;
            let yscale = canvasHeight / occupyHeight;
            let scale = Math.min(xscale, yscale);
            let dx = (canvasWidth - occupyWidth * scale)/2 - leftbound*scale + canvasX;
            let dy = (canvasHeight - occupyHeight * scale)/2 - upperbound*scale + canvasY;
            //console.log("canvasWidth, canvasHeight, occupyWidth, occupyHeight", canvasWidth,canvasHeight, occupyWidth,occupyHeight);
            let calTransform = {
                "k": scale,
                "x":dx,
                "y":dy

            }
            return calTransform;

        }


        // ---- The above code is reserved for overview canvas calculation. NOT IMPORTANT ---- //

        // 2. Main render function.
        function renderContext(context:any){


            // Unfocused nodes rendering.
            if(DisplayUnfocusedNodes){
                tempData.links.filter((d:any)=>{
                    if(d.source.highlight && d.target.highlight){
                        return false;
                    }else{
                        return true;
                    }
                }).forEach(function(d:any) {
                    // Draw Line
                    drawLine(context, d.color, d.source.x, d.source.y, d.target.x, d.target.y, null, d.weight);
                });
        
                // Draw the nodes
                tempData.nodes.filter((d:any)=>{
                    return !d["highlight"];
                }
                ).forEach(function(d:any, i:any) {
                    // Draw Node Glyph
                    //console.log("radius",d.radius);
                    let node_inner_radius = d.radius - radius_gap;
                    let node_radius = d.radius;
                    let node_outer_radius = d.radius * 2;
                    let node_outer_arc_encoded_value = d.node_weight;
                    let node_outer_arc_radius = node_outer_radius + radius_gap * 5;
                    drawNodeGlyph(context, d.color, node_inner_radius, 
                        node_radius, node_outer_radius, d.x, d.y, false,
                        node_outer_arc_encoded_value, node_outer_arc_radius);
                });
            }
            
            // Focused nodes rendering
            tempData.links.filter((d:any)=>{
                if(d.source.highlight && d.target.highlight){
                    return true;
                }else{
                    return false;
                }
            }).forEach(function(d:any) {
                drawLine(context, d.color, d.source.x, d.source.y, d.target.x, d.target.y, 5 * d.weight, d.weight);
            });
            tempData.nodes.filter((d:any)=>{
                return d["highlight"];
                
            }).forEach(function(d:any,i:any){
                let node_inner_radius = d.radius - radius_gap;
                let node_radius = d.radius;
                let node_outer_radius = d.radius * 2;
                let node_outer_arc_encoded_value = d.node_weight;
                let node_outer_arc_radius = node_outer_radius + radius_gap * 5;
                drawNodeGlyph(context, d.color, node_inner_radius, 
                    node_radius, node_outer_radius, d.x, d.y, false,
                    node_outer_arc_encoded_value, node_outer_arc_radius);

            })
        }

        // Main Function for Updating Layout.
        function simulationUpdate(){
            context.save();
            context.clearRect(0, 0, graphWidth, height);
            context.translate(transform.x, transform.y);
            context.scale(transform.k, transform.k);
            renderContext(context);
            context.restore();
            
            //let canvasWidth = 100 * graphWidth / height;
            // ---- The following code is reserved for overview canvas calculation. NOT IMPORTANT ---- //

            if(DisplayOverview){
                let graph_bounding_box = calculateGraphBoundingBox();
                calTransform = calculateTransform(canvasX, canvasY, canvasWidth, canvasHeight, graph_bounding_box);
                let rect_configuration = {
                    "x":0, "y":0, "width":graphWidth, "height":height
                }
                let overview_configuration = {
                    "x":canvasX,
                    "y":canvasY,
                    "width":canvasWidth,
                    "height":canvasHeight
                }
                let overview_inverse_rect = rectInverseTransform(overview_configuration, calTransform);
                let overview_bounding_box = {
                    "leftbound":overview_inverse_rect["x"],
                    "upperbound":overview_inverse_rect["y"],
                    "occupyWidth":overview_inverse_rect["width"],
                    "occupyHeight":overview_inverse_rect["height"]
                }
                let view_inverse_configuration = rectInverseTransformAndClip(rect_configuration, transform, overview_bounding_box);
                let view_configuration = rectTransform(view_inverse_configuration, calTransform); 
    
                overview_context.save();
                overview_context.clearRect(0, 0, graphWidth, height);
                drawRectStroke(overview_context, canvasX, canvasY, canvasWidth, canvasHeight);
                drawRect(overview_context, canvasX, canvasY, canvasWidth, canvasHeight);
                
                overview_context.translate(calTransform.x, calTransform.y);
                overview_context.scale(calTransform.k, calTransform.k);
                renderContext(overview_context);
                overview_context.scale(1/calTransform.k, 1/calTransform.k);
                overview_context.translate(-calTransform.x, -calTransform.y);
                drawRectStroke(overview_context, view_configuration["x"], view_configuration["y"], view_configuration["width"], view_configuration["height"],"#000");
                drawRect(overview_context, view_configuration["x"], view_configuration["y"], view_configuration["width"], view_configuration["height"],"#ccc",0.5);
                overview_context.restore();
                
            }
            // ---- The above code is reserved for overview canvas calculation. NOT IMPORTANT ---- //

            
            handleMouseMove(middleCanvas, true);
        }


        // When hovering, changing middle canvas.
        function middleCanvasSimulationUpdate(){
            let judgeHovered = (d:any)=>{
                if(d.hasOwnProperty("hovered") && d["hovered"]){
                    return true;
                }else{
                    return false;
                }
            }
            middle_context.save();
            
            middle_context.clearRect(0, 0, graphWidth, height);
            middle_context.translate(transform.x, transform.y);
            middle_context.scale(transform.k, transform.k);
            tempData.links.filter((d:any)=>{
                if(judgeHovered(d.source) && judgeHovered(d.target)){
                    return true;
                }else{
                    return false;
                }
            }).forEach(function(d:any) {
                drawLine(middle_context, d.real_color, d.source.x, d.source.y, d.target.x, d.target.y, null, d.weight);
            });
            // Draw the hovered nodes
            tempData.nodes.filter((d:any)=>{
                return judgeHovered(d);
            }).sort(order_determine).forEach(function(d:any, i:any) {
                let node_inner_radius = d.radius - radius_gap;
                let node_radius = d.radius;
                let node_outer_radius = d.radius * 2;
                let node_outer_arc_encoded_value = d.node_weight;
                let node_outer_arc_radius = node_outer_radius + radius_gap * 5;
                drawNodeGlyph(middle_context, d.real_color, node_inner_radius*d.hover_cons, 
                    node_radius*d.hover_cons, node_outer_radius*d.hover_cons, d.x, d.y, true, 
                    node_outer_arc_encoded_value, node_outer_arc_radius*d.hover_cons, false);
            });
            middle_context.restore();
        }
     }
 
    public render() {     
        return (
            <div id="force_directed_graph">
                <canvas id="bottom" className="AbsPos" />
                <canvas id="middle" className="AbsPos"/>
                <canvas id="overview" className="AbsPos"/>
                <svg
                    id="svgChart"
                    xmlns="http://www.w3.org/2000/svg"
                    className="AbsPos"
                >
                    <g id="ForceDirectedLegend">

                    </g>
                    <g id="ForceDirectedColorLegend">

                    </g>
                </svg>
                <div id="tooltip" className="AbsPos" />
                
                <canvas id="event" className="AbsPos"/>
            </div>
            

        )

    }
}


// Color Helper

const d3_10color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
const color_brewer1 = ["#fbb4ae","#b3cde3", "#ccebc5","#decbe4","#fed9a6","#ffffcc","#e5d8bd","#fddaec"];
const color_brewer2 = ["#b3e2cd","#fdcdac", "#cbd5e8", "#f4cae4", "#e6f5c9", "#fff2ae", "#f1e2cc", "#cccccc"];
const COLORS: string[] = [
    "#1A7AB1",
    "#ADC8E6",
    "#FF772D",
    "#FFB87F",
    "#2AA13A",
    "#98E090",
    "#FF9398",
    "#9467B9",
    "#C5B0D3",
    "#C49B95",
    "#E474C0",
    "#F7B4D1",
    "#BCBC3D",
    "#07C1CD"
    ]

const GREEN: string[] = [
    "#498B77",
    "#89C2AE",
    "#C1D6D3"
]
const BLUE: string[] = [
    "#3E97C7",
    "#72B3CF",
    "#8FCCDD",
    "#C8DADE"

]
const ORANGE: string[] = [
    "#E96206",
    "#F79143",
    "#F6AD76",
    "#F7CEA7"
]
const PINK: string[] = [
    "#F6B1C3",
    "#F07F93",
    "#DE4863",
    "#BC0F46"

]
const RED: string[] = ["#DC143C"];
const YELLOW : string[] = ['#fee08b'];
const GRAY: string[] = ['#999999'];
const getLinearColor = (ColorList: string[], step:number) => {
    let totalColor = ColorList.length;
    let divide = 1/ (totalColor - 1);
    let location = Math.floor(step / divide);
    if(location == totalColor - 1){
        location = location - 1;
    }
    let offset = step - location * divide;
    let adjusted_offset = offset / divide;
    return getGradientColor(ColorList[location], ColorList[location+1], adjusted_offset);
}
const getGradientColor = (startColor : string,endColor :string,step : number) => {
    let colorRgb = (sColor : string)=>{
        var reg = /^#([0-9a-fA-f]{3}|[0-9a-fA-f]{6})$/;
        var sColor = sColor.toLowerCase();
        if(sColor && reg.test(sColor)){
            if(sColor.length === 4){
                var sColorNew = "#";
                for(var i=1; i<4; i+=1){
                    sColorNew += sColor.slice(i,i+1).concat(sColor.slice(i,i+1));
                }
                sColor = sColorNew;
            }
            var sColorChange = [];
            for(var i=1; i<7; i+=2){
                sColorChange.push(parseInt("0x"+sColor.slice(i,i+2)));
            }
            return sColorChange;
        }else{
            return sColor;
        }
    };
    startColor = startColor.replace(/\s+/g,"");
    endColor = endColor.replace(/\s+/g,"");
    let startRGB : any = colorRgb(startColor);//转换为rgb数组模式
    //console.log(startRGB);
    let startR = startRGB[0];
    let startG = startRGB[1];
    let startB = startRGB[2];

    let endRGB : any = colorRgb(endColor);
    //console.log(endRGB);

    let endR = endRGB[0];
    let endG = endRGB[1];
    let endB = endRGB[2];
    if(step>1){
        console.log("out of range step: ", step);
        step = 1;
    }else if(step<0){
        console.log("out of range step: ", step);
        step = 0;
    }
    let sR = (endR-startR)*step;//总差值
    let sG = (endG-startG)*step;
    let sB = (endB-startB)*step;
    var R = parseInt((sR+startR));
    var G = parseInt((sG+startG));
    var B = parseInt((sB+startB));
    var strHex = "#";
    var aColor = new Array();
    aColor[0] = R;
    aColor[1] = G;
    aColor[2] = B;
    for(let j=0; j<3; j++){
        let hex : string = Number(aColor[j]).toString(16);
        let shex : string = Number(aColor[j])<10 ? '0'+hex :hex;
        if(shex === "0"){
            shex += shex;
        }
        strHex += shex;
    }
    return strHex;
}


const EChartsColor = [
    "#c23531",
    "#2f4554",
    "#61a0a8",
    "#d48265",
    "#91c7ae",
    "#749f83"
]

const DefaultColor = BLUE[1];
const StartColor = BLUE[0];
const EndColor = RED[0];
const getNodeColor = ( node_label:number,color_encode:number = 2) =>{
    if(color_encode === 1 || color_encode === 2 || color_encode === 3){
        return d3_10color[node_label];
    }else if(color_encode === 5){
        if(node_label){
            return GREEN[0];
        }else{
            return RED[0];
        }
    }
    
}

// Transform Data Helper

// Construct Graph In from Graph obj
function constructGraphIn(graph_obj:any){
    let senders = graph_obj.srcs;
    let receivers = graph_obj.dsts;
    let num_nodes = graph_obj.num_nodes;
    return {
        "senders":senders,
        "receivers":receivers,
        "num_nodes":num_nodes
    }
}


// Construct Neighbor Set from Graph in. 
function constructNeighborSet(graph_in:any){
    // Input: Graph_in
    //   senders:  source of edges.
    //   receivers:   target of edges.
    //   node_num:  number of nodes.
    // Output: 
    //   NeighborSet: Dict[Key] <-- incoming neighbors.
    let senders = graph_in.senders;
    let receivers = graph_in.receivers;
    let node_num = graph_in.num_nodes;
    let NeighborSet:any = {};
    for(let i = 0; i<node_num ;i++){
        NeighborSet[i] = [];
    }
    for(let i = 0; i< receivers.length; i++){
        let nowreceiver = receivers[i];
        if(nowreceiver in NeighborSet){
        }else{
            NeighborSet[nowreceiver] = []
        }
        NeighborSet[nowreceiver].push(senders[i]);
    }
    return NeighborSet;
}
function constructPathDict(message_passing:any){
    let senders = message_passing.senders;
    let receivers = message_passing.receivers;
    let values = message_passing.values;
    let PathDict:any = {};
    for(let i = 0; i< receivers.length; i++){
        let nowreceiver = receivers[i];
        if(nowreceiver in PathDict){
        }else{
            PathDict[nowreceiver] = {}
        }
        PathDict[nowreceiver][senders[i]] = values[i];
    }
    return PathDict;
}
function getTrainColor(node_id:any, train_set:any){
    if(train_set.has(node_id)){
        //return "#fff";
        return "#000";
    }else{
        return "#fff";
    }
}
function getNodeStatisticStr(selectedNodeLength: number, totalNodeLength: number){
    let str : string = "" + selectedNodeLength + "/"+ totalNodeLength;
    let percentage : number ;
    if(totalNodeLength === 0){

    }else{
        percentage = selectedNodeLength / totalNodeLength * 100;
        str = str + " (" + percentage.toFixed(2) +"%)"
    }
    return str;
}


function compareSelectedNodeIdList(list_a:any, list_b:any){
    if(list_a.length === list_b.length){
        for(let i = 0; i<list_a.length; i++){
            if(list_a[i] === list_b[i]){

            }else{
                return false;
            }
        }
        return true;
    }else{
        return false;
    }
}

function get_boundingbox(graph_layout:any[]){
    if(graph_layout.length === 0){
        return {
            "xmin":0,
            "xmax":0,
            "ymin":0,
            "ymax":0
        }
    }else{
        let xmin = graph_layout[0][0];
        let xmax = graph_layout[0][0];
        let ymin = graph_layout[0][1];
        let ymax = graph_layout[0][1];
        for(let i = 0; i< graph_layout.length; i++){
            let nowx = graph_layout[i][0];
            let nowy = graph_layout[i][1];
            if(xmin > nowx){
                xmin = nowx;
            }
            if(xmax < nowx){
                xmax = nowx;
            }
            if(ymin > nowy){
                ymin = nowy;
            }
            if(ymax < nowy){
                ymax = nowy;
            }
        }
        return {
            "xmin":xmin,
            "xmax":xmax,
            "ymin":ymin,
            "ymax":ymax
        }
    }
}
function transform_graphlayout(graph_layout:any[], width:number, height:number){
    if(graph_layout.length === 0){
        return graph_layout;
    }else{
        let bounding_box = get_boundingbox(graph_layout);
        //let canvas_centerx = 300;
        //let canvas_centery = 300;
        //let width = Swidth;
        //let height = Sheight;
        let margin = 20;
        if(graph_layout.length >= 100){
            margin = 20;
        }
        
        let realwidth = width - 2*margin;
        let realheight = height - 2*margin;
        let gap_x = bounding_box["xmax"] - bounding_box["xmin"];
        let gap_y = bounding_box["ymax"] - bounding_box["ymin"];
        if(gap_x === 0){
            gap_x = 1e-16;
        }
        if(gap_y === 0){
            gap_y = 1e-16;
        }
        let realscale = Math.min(realwidth / gap_x, realheight / gap_y);
        let left = margin + (realwidth - realscale * gap_x) / 2;
        let top = margin + (realheight - realscale * gap_y) / 2; 
        let xmin = bounding_box["xmin"];
        let ymin = bounding_box["ymin"];
        let new_graph_layout = [];
        for(let i = 0; i< graph_layout.length; i++){
            let nowx = graph_layout[i][0];
            let nowy = graph_layout[i][1];
            let locx = left+ (nowx - xmin) * realscale;
            let locy = top + (nowy - ymin) * realscale;
            new_graph_layout.push([locx,locy]);
        }
        return new_graph_layout;
    }
}
function skew_weight(weight:any, range_min:any=0.1, range_max:any=1){
    // Assume weight is [0,1]
    return (weight - 0) * 0.9 + range_min;
}



export { RED,YELLOW, EChartsColor,   getNodeColor, constructNeighborSet,  constructPathDict,getTrainColor, 
     compareSelectedNodeIdList,getNodeStatisticStr,constructGraphIn,transform_graphlayout,skew_weight }

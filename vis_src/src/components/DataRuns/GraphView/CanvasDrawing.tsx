import { color } from "d3";

// Canvas Handle Function
function drawRectStroke(context:any, x:any, y:any, width:any, height:any, strokeColor:any="#bbb"){
    context.beginPath();
    context.strokeStyle = strokeColor;
    context.rect(x, y, width, height);
    context.stroke();
}
function drawRect(context:any, x:any, y:any, width:any, height:any, fillColor:any="#fff", opacity:any=0.8){
    context.fillStyle = fillColor;
    context.globalAlpha = opacity;
    context.fillRect(x, y, width, height);
    context.globalAlpha = 1.0;
}
function drawCircleStroke(context:any, color:any, radius:any, x:any, y:any, lineWidth:number){
    context.lineWidth = lineWidth
    context.strokeStyle = color;
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI, true);
    context.stroke();
}

function drawCircle(context:any, color:any, radius:any, x:any, y:any, alpha:any=1){
    let original_globalAlpha = context.globalAlpha;
    context.globalAlpha = alpha;
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI, true);
    context.fillStyle = color;
    context.fill();
    context.globalAlpha = original_globalAlpha;

}

function drawOnePie(context:any, color:any, radius:any, x:any, y:any, startAngle:any, endAngle:any, alpha:any=1){
    let original_globalAlpha = context.globalAlpha;
    context.globalAlpha = alpha;
    context.beginPath();
    context.moveTo(x,y);
    context.arc(x, y, radius, startAngle, endAngle);
    context.fillStyle = color;
    context.closePath();
    context.fill();
    context.globalAlpha = original_globalAlpha;

}
function drawOneArc(context:any, color:any, radius:any, x:any, y:any, startAngle:any, endAngle:any){
    context.moveTo(x,y);
    context.beginPath();
    context.arc(x, y, radius, startAngle, endAngle);
    context.strokeStyle = color;
    context.stroke();
}
/*
//Backup version
function drawNodeGlyph(context:any, colorlist:any, inner_radius:any, radius:any, outer_radius:any, x:any, y:any, enableStroke:boolean=false){
    
    drawCircle(context, colorlist[4], outer_radius, x, y);
    if(enableStroke){
        drawCircleStroke(context, "#000", outer_radius, x, y, 2);
    }
    drawOnePie(context, colorlist[1], outer_radius, x, y, (-150)/180*Math.PI, (-30)/180*Math.PI);
    drawOnePie(context, colorlist[2], outer_radius, x, y, (-30)/180*Math.PI, (+90)/180*Math.PI);
    drawOnePie(context, colorlist[3], outer_radius, x, y, (+90)/180*Math.PI, (+210)/180*Math.PI);
    for(let i = 0; i<3; i++){
        let angle = (-150+120*i)/180*Math.PI;
        let x1 = x + radius*Math.cos(angle);
        let y1 = y + radius*Math.sin(angle);
        let x2 = x + outer_radius*Math.cos(angle);
        let y2 = y + outer_radius*Math.sin(angle);
        // drawLine(context, "#ddd", x1, y1, x2 ,y2, 0.5);
        drawLine(context, colorlist[4], x1, y1, x2 ,y2, radius-inner_radius);
    }
    drawCircle(context, colorlist[4], radius, x, y);
    drawCircle(context, colorlist[0], inner_radius, x, y);
}*/

/*
function drawNodeGlyph(context:any, colorlist:any, inner_radius:any, radius:any, outer_radius:any, x:any, y:any, 
    enableStroke:boolean=false, outer_arc_encoded_value:any=0.5, outer_arc_radius:any=2, enable_alpha_mode=true){
    let value = outer_arc_encoded_value;
    //let original_globalAlpha = context.globalAlpha;
    if(value<0) value = 0;
    else if(value>1) value = 1;
    let alpha = 1;
    if(enable_alpha_mode){
        alpha = value;
    }
    
    //context.globalAlpha = value;
    drawCircle(context, colorlist[4], outer_radius, x, y, alpha);
    if(enableStroke){
        drawCircleStroke(context, "#000", outer_radius, x, y, 2);
    }
    
    drawOnePie(context, colorlist[1], outer_radius, x, y, (-150)/180*Math.PI, (-30)/180*Math.PI, alpha);
    drawOnePie(context, colorlist[2], outer_radius, x, y, (-30)/180*Math.PI, (+90)/180*Math.PI, alpha);
    drawOnePie(context, colorlist[3], outer_radius, x, y, (+90)/180*Math.PI, (+210)/180*Math.PI, alpha);
    for(let i = 0; i<3; i++){
        let angle = (-150+120*i)/180*Math.PI;
        let x1 = x + radius*Math.cos(angle);
        let y1 = y + radius*Math.sin(angle);
        let x2 = x + outer_radius*Math.cos(angle);
        let y2 = y + outer_radius*Math.sin(angle);
        // drawLine(context, "#ddd", x1, y1, x2 ,y2, 0.5);
        drawLine(context, colorlist[4], x1, y1, x2 ,y2, radius-inner_radius, alpha);
    }
    drawCircle(context, colorlist[4], radius, x, y, alpha);
    drawCircle(context, colorlist[0], inner_radius, x, y, alpha);
    //context.globalAlpha = original_globalAlpha;

    //let degree = 360 * value - 90;
    //drawOneArc(context, colorlist[5], outer_arc_radius, x, y, (-90)/180*Math.PI, degree/180*Math.PI);
}

function drawNodeGlyph_v1(context:any, colorlist:any, inner_radius:any, radius:any, outer_radius:any, x:any, y:any, 
    enableStroke:boolean=false, outer_arc_encoded_value:any=0.5, outer_arc_radius:any=2, enable_alpha_mode=true){
    let value = outer_arc_encoded_value;
    if(value<0) value = 0;
    else if(value>1) value = 1;
    let alpha = 1;
    if(enable_alpha_mode){
        alpha = value;
    }
    if(enableStroke){
        drawCircleStroke(context, "#000", outer_radius, x, y, 2);
    }
    drawCircle(context, colorlist[0], outer_radius, x, y, alpha);
}
function drawNodeGlyph(context:any, colorlist:any, inner_radius:any, radius:any, outer_radius:any, x:any, y:any, 
    enableStroke:boolean=false, outer_arc_encoded_value:any=0.5, outer_arc_radius:any=2, enable_alpha_mode=true){
    let value = outer_arc_encoded_value;
    //let original_globalAlpha = context.globalAlpha;
    if(value<0) value = 0;
    else if(value>1) value = 1;
    let alpha = 1;
    if(enable_alpha_mode){
        alpha = value;
    }
    
    //context.globalAlpha = value;
    // Background circle
    
    
    
    // Pie chart drawing
    let length_model = colorlist.length - 1;
    if(length_model > 0){
        drawCircle(context, "#fff", outer_radius, x, y, alpha);
        if(enableStroke){
            drawCircleStroke(context, "#000", outer_radius, x, y, 2);
        }
        let step_angle = 360 / length_model;
        let current_angle = -90 - step_angle / 2; 
        for(let i = 1; i<colorlist.length; i++){
            let start_angle = current_angle;
            let end_angle = start_angle + step_angle;
            drawOnePie(context, colorlist[i], outer_radius, x, y, (start_angle)/180*Math.PI, (end_angle)/180*Math.PI, alpha);
            current_angle = end_angle;
        }
        if(length_model > 1){
            current_angle = -90 - step_angle / 2;
            for(let i = 1; i<colorlist.length; i++){
                let angle = (current_angle)/180*Math.PI;
                let x1 = x + radius*Math.cos(angle);
                let y1 = y + radius*Math.sin(angle);
                let x2 = x + outer_radius*Math.cos(angle);
                let y2 = y + outer_radius*Math.sin(angle);
                // drawLine(context, "#ddd", x1, y1, x2 ,y2, 0.5);
                drawLine(context, "#fff", x1, y1, x2 ,y2, radius-inner_radius, alpha);
                current_angle = current_angle + step_angle;
            }
        }
        drawCircle(context, "#fff", radius, x, y, alpha);
        drawCircle(context, colorlist[0], inner_radius, x, y, alpha);

    }else{
        drawCircle(context, colorlist[0], outer_radius, x, y, alpha);
        if(enableStroke){
            drawCircleStroke(context, "#000", outer_radius, x, y, 2);
        }
        //drawCircle(context, colorlist[0], radius, x, y, alpha);
        //drawCircle(context, colorlist[0], inner_radius, x, y, alpha);
    }
    

*/

function drawNodeGlyph(context:any, colorlist:any, inner_radius:any, radius:any, outer_radius:any, x:any, y:any, 
    enableStroke:boolean=false, outer_arc_encoded_value:any=0.5, outer_arc_radius:any=2, enable_alpha_mode=true){
    let value = outer_arc_encoded_value;
    //let original_globalAlpha = context.globalAlpha;
    if(value<0) value = 0;
    else if(value>1) value = 1;
    let alpha = 1;
    if(enable_alpha_mode){
        alpha = value;
    }
    
    //context.globalAlpha = value;
    // Background circle
    
    
    
    // Pie chart drawing
    let length_model = colorlist.length - 1;
    if(length_model > 0){
        drawCircle(context, "#fff", outer_radius, x, y, alpha);
        if(enableStroke){
            drawCircleStroke(context, "#000", outer_radius, x, y, 2);
        }
        let step_angle = 360 / length_model;
        let current_angle = -90 - step_angle / 2; 
        for(let i = 1; i<colorlist.length; i++){
            let start_angle = current_angle;
            let end_angle = start_angle + step_angle;
            drawOnePie(context, colorlist[i], outer_radius, x, y, (start_angle)/180*Math.PI, (end_angle)/180*Math.PI, alpha);
            current_angle = end_angle;
        }
        if(length_model > 1){
            current_angle = -90 - step_angle / 2;
            for(let i = 1; i<colorlist.length; i++){
                let angle = (current_angle)/180*Math.PI;
                let x1 = x + radius*Math.cos(angle);
                let y1 = y + radius*Math.sin(angle);
                let x2 = x + outer_radius*Math.cos(angle);
                let y2 = y + outer_radius*Math.sin(angle);
                // drawLine(context, "#ddd", x1, y1, x2 ,y2, 0.5);
                drawLine(context, "#fff", x1, y1, x2 ,y2, radius-inner_radius, alpha);
                current_angle = current_angle + step_angle;
            }
        }
        drawCircle(context, "#fff", radius, x, y, alpha);
        drawCircle(context, colorlist[0], inner_radius, x, y, alpha);

    }else{
        drawCircle(context, colorlist[0], outer_radius, x, y, alpha);
        if(enableStroke){
            drawCircleStroke(context, "#000", outer_radius, x, y, 2);
        }
        //drawCircle(context, colorlist[0], radius, x, y, alpha);
        //drawCircle(context, colorlist[0], inner_radius, x, y, alpha);
    }
    

    
    //context.globalAlpha = original_globalAlpha;

    //let degree = 360 * value - 90;
    //drawOneArc(context, colorlist[5], outer_arc_radius, x, y, (-90)/180*Math.PI, degree/180*Math.PI);
}
function drawLine(context:any, color:any, x1:any, y1:any, x2:any, y2:any, linewidth:any=null, weight:any=1){
    let original_globalAlpha = context.globalAlpha;
    let value = weight;
    if(value<0) value = 0;
    else if(value>1) value = 1;
    context.globalAlpha = value;
    context.strokeStyle = color;
    if(linewidth){
        context.lineWidth = linewidth;
    }
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
    context.globalAlpha = original_globalAlpha;
}

export {drawRectStroke, drawRect, drawCircleStroke, 
    drawCircle, drawOnePie, drawOneArc, drawNodeGlyph, drawLine }

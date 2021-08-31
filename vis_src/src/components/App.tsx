import { Col, Layout, Row } from 'antd';

import React from 'react';
import logo from './logo2.png';
import './App.css';
import DataRunsContainer from '../container/DataRunsContainer';
// import DataView from "./DataView";
//import SidePanel from '../components/SidePanel';
const { Content, Header } = Layout;
export interface IProps{

}
export interface IState {
    contentWidth:number,
    contentHeight:number,
    screenWidth:number,
    screenHeight:number
}
class App extends React.Component<IProps, IState> {
  public ContentRef:any;
  constructor(props:IProps) {
      super(props);
      this.ContentRef = React.createRef();
      this.onResize = this.onResize.bind(this);
      this.state = {
          contentWidth : 0,
          contentHeight: 0,
          screenHeight: 0,
          screenWidth :0
      }

  }
  public getLayoutConfig(){
    let contentWidth:number = 0;
    let contentHeight:number = 0;
    if(this.ContentRef){
      contentWidth = this.ContentRef.current.offsetWidth;
      contentHeight = this.ContentRef.current.offsetHeight;
    }
    return {
      contentWidth:contentWidth, 
      contentHeight:contentHeight
    }
  }
  public onResize(){
    this.updateLayoutState();
 }  
 public updateLayoutState(){
    let contentLayout = this.getLayoutConfig();
    let contentWidth = contentLayout.contentWidth;
    let contentHeight = contentLayout.contentHeight;
    //console.log("contentWidth, height", contentWidth, contentHeight)
   this.setState({
      contentWidth:contentWidth,
      contentHeight:contentHeight,
       screenHeight: window.innerHeight,
       screenWidth: window.innerWidth
   })
 }
 componentDidMount(){
    window.addEventListener('resize', this.onResize)
    this.updateLayoutState();
 }
 componentDidUpdate(prevProps:IProps, prevState:IState) {
  // if(prevState.contentWidth!==this.state.contentWidth || )
  //this.updateLayoutState();
 }
  public render() {
    let {screenWidth, screenHeight, contentWidth, contentHeight} = this.state;
    return (
      <Layout className="app" >
          <Header className='appHeader'>
          GNNVis
                  <img src={logo} className='appLogo' alt-text="logo"/>
          </Header>
          <Content className='appContent' >
              <div style={{ "height": "100%", "width":"100%" }} ref={this.ContentRef}>
                {(contentWidth>0 && contentHeight >0)?<DataRunsContainer contentWidth={contentWidth} contentHeight={contentHeight}/>:<div />}
              </div>
              <div id="tooltip_proj" />
              <div id="tooltip_matrix" />
          </Content>
      </Layout>
    );
  }
}

export default App;

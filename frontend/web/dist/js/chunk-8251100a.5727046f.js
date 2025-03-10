(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-8251100a"],{2023:function(e,t,a){},"32e9":function(e,t,a){"use strict";a("4f6b")},"4f6b":function(e,t,a){},"891c":function(e,t,a){"use strict";a.r(t);var l=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"experiment-container"},[a("div",{staticClass:"top-box"},[a("el-button",{attrs:{type:"primary",icon:"el-icon-plus"},on:{click:e.openDialog}},[e._v("创建实验")])],1),a("div",{staticClass:"bottom-box"},[a("el-row",{attrs:{gutter:20}},e._l(e.tasks,(function(t){return a("el-col",{key:t.id,attrs:{span:6}},[a("el-card",{class:["box-card","box-card-"+t.experiment_status]},[a("div",{staticClass:"card-title",attrs:{slot:"header"},slot:"header"},[a("span",[e._v(e._s(t.experiment_name))]),a("span",{staticClass:"card-header-right"},[a("el-tooltip",{staticClass:"item",attrs:{content:"查看实验"}},[a("i",{staticClass:"el-icon-info",on:{click:function(a){return e.info(t)}}})]),a("el-tooltip",{staticClass:"item",attrs:{content:"删除实验"}},[a("i",{staticClass:"el-icon-delete",on:{click:function(a){return e.del(t)}}})])],1)]),a("div",{staticClass:"exp-item"},[a("span",{staticClass:"exp-item-label"},[e._v("任务类型：")]),a("span",{staticClass:"exp-item-value"},[e._v(e._s(t.task_type))])]),a("div",{staticClass:"exp-item"},[a("span",{staticClass:"exp-item-label"},[e._v("模型类型：")]),a("span",{staticClass:"exp-item-value"},[e._v(e._s(t.model_type))])]),a("div",{staticClass:"exp-item"},[a("span",{staticClass:"exp-item-label"},[e._v("描述信息：")]),a("span",{staticClass:"exp-item-value"},[e._v(e._s(t.task_desc))])]),a("div",{staticClass:"exp-item"},[a("span",{staticClass:"exp-item-label"},[e._v("状态：")]),a("span",{staticClass:"exp-item-value"},[e._v(e._s(t.experiment_status))])])])],1)})),1)],1),a("el-dialog",{staticClass:"experiment-dialog",attrs:{title:"创建实验",visible:e.dialogFormVisible,top:"10vh","close-on-click-modal":!1},on:{"update:visible":function(t){e.dialogFormVisible=t}}},[e.dialogFormVisible?a("ExperimentForm",{ref:"experimentForm"}):e._e(),a("div",{staticClass:"dialog-footer",attrs:{slot:"footer"},slot:"footer"},[a("el-button",{on:{click:function(t){e.dialogFormVisible=!1}}},[e._v("取 消")]),a("el-button",{attrs:{type:"primary"},on:{click:e.openExperiment}},[e._v("开启实验")])],1)],1)],1)},o=[],s=(a("e9c4"),a("b64b"),a("d3b7"),a("159b"),function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",[a("el-form",{ref:"form",staticClass:"experiment-form",attrs:{model:e.form,rules:e.rules,"label-width":"100px","label-position":"top"}},[a("el-row",{attrs:{gutter:20}},[a("el-col",{attrs:{span:12}},[a("el-form-item",{attrs:{label:"实验名称",prop:"experiment_name"}},[a("el-input",{model:{value:e.form.experiment_name,callback:function(t){e.$set(e.form,"experiment_name",t)},expression:"form.experiment_name"}})],1)],1),a("el-col",{attrs:{span:12}},[a("el-form-item",{attrs:{label:"任务类型",prop:"task_type"}},[a("el-select",{attrs:{clearable:"",placeholder:"请选择任务类型"},model:{value:e.form.task_type,callback:function(t){e.$set(e.form,"task_type",t)},expression:"form.task_type"}},e._l(e.taskList,(function(e,t){return a("el-option",{key:t,attrs:{label:e.label,value:e.value}})})),1)],1)],1),a("el-col",{attrs:{span:24}},[a("el-form-item",{attrs:{label:"任务描述",prop:"task_desc"}},[a("el-input",{attrs:{type:"textarea",maxlength:"155","show-word-limit":""},model:{value:e.form.task_desc,callback:function(t){e.$set(e.form,"task_desc",t)},expression:"form.task_desc"}})],1)],1),a("el-col",{attrs:{span:24}},[a("el-form-item",{attrs:{label:"基础模型",prop:"model_type"}},[a("el-row",[a("el-col",{attrs:{span:12}},[a("el-input",{model:{value:e.form.model_type,callback:function(t){e.$set(e.form,"model_type",t)},expression:"form.model_type"}})],1),a("el-col",{attrs:{span:10,offset:1}},[a("el-button",{attrs:{type:"text"},on:{click:e.suggest}},[e._v("推荐")])],1)],1)],1)],1),a("el-col",{attrs:{span:24}},[a("el-form-item",{attrs:{label:"训练数据",prop:"files"}},[a("el-button",{on:{click:e.upload}},[e._v("上传")]),a("input",{attrs:{hidden:"",id:"upload-file",type:"file",name:"files",accept:".csv"},on:{change:e.onFileChanged}}),a("input",{attrs:{hidden:"",type:"file",id:"upload-folder",name:"folderUpload",webkitdirectory:"",multiple:""},on:{change:e.onFolderChanged}}),e.uploadNum?a("span",{staticClass:"uploadClass"},[e._v("已上传"+e._s(e.uploadNum)+"个文件"),a("span",{staticClass:"tagClose",on:{click:e.handleUploadClose}},[e._v("x")])]):e._e()],1)],1)],1),a("el-row",{attrs:{gutter:20}},[a("el-col",{attrs:{span:12}},[a("el-form-item",{attrs:{label:"调优算法",prop:"tp_tuner"}},[a("el-select",{attrs:{clearable:"",placeholder:"请选择超参数调优算法"},model:{value:e.form.tp_tuner,callback:function(t){e.$set(e.form,"tp_tuner",t)},expression:"form.tp_tuner"}},e._l(e.tp_tunerList,(function(e,t){return a("el-option",{key:t,attrs:{label:e.label,value:e.value}})})),1)],1)],1),a("el-col",{attrs:{span:12}},[a("el-form-item",{attrs:{label:"最大试验次数",prop:"tp_max_trials"}},[a("el-input",{model:{value:e.form.tp_max_trials,callback:function(t){e.$set(e.form,"tp_max_trials",t)},expression:"form.tp_max_trials"}})],1)],1)],1),a("el-row",{attrs:{gutter:20}},[a("el-col",{attrs:{span:24}},[a("el-form-item",{attrs:{label:"训练参数",prop:"training_params"}},[a("el-collapse",{attrs:{accordion:""},on:{change:e.handleChange},model:{value:e.activeNames,callback:function(t){e.activeNames=t},expression:"activeNames"}},[a("el-collapse-item",{attrs:{title:"展开查看",name:"1"}},[a("json-editor",{attrs:{mode:"code"},model:{value:e.form.training_params,callback:function(t){e.$set(e.form,"training_params",t)},expression:"form.training_params"}})],1)],1)],1)],1)],1),a("el-dialog",{attrs:{title:"提示",visible:e.dialogVisible,width:"50%","before-close":e.handleClose,"append-to-body":""},on:{"update:visible":function(t){e.dialogVisible=t}}},[a("el-row",{attrs:{gutter:20}},[a("el-col",{attrs:{span:24}},[a("el-form-item",{attrs:{label:"基础模型",prop:"model_foundation"}},[a("el-input",{model:{value:e.form.model_foundation,callback:function(t){e.$set(e.form,"model_foundation",t)},expression:"form.model_foundation"}})],1)],1),a("el-col",{attrs:{span:24}},[a("el-form-item",{attrs:{label:"推荐原因",prop:"recommend"}},[a("el-input",{attrs:{type:"textarea",rows:4},model:{value:e.form.recommend,callback:function(t){e.$set(e.form,"recommend",t)},expression:"form.recommend"}})],1)],1),a("el-col",{attrs:{span:24}},[a("el-form-item",{attrs:{label:"训练参数",prop:"training_params"}},[a("json-editor",{attrs:{mode:"code"},model:{value:e.form.training_params,callback:function(t){e.$set(e.form,"training_params",t)},expression:"form.training_params"}})],1)],1)],1),a("span",{staticClass:"dialog-footer",attrs:{slot:"footer"},slot:"footer"},[a("el-button",{on:{click:function(t){e.dialogVisible=!1}}},[e._v("取 消")]),a("el-button",{attrs:{type:"primary"},on:{click:e.recommend}},[e._v("确 定")])],1)],1)],1)],1)}),i=[],n=(a("caad"),a("c114")),r={data:function(){return{loading:!1,dialogVisible:!1,formLabelWidth:"120px",uploadNum:null,uploadParameters:null,tagShutDown:!1,tag:null,form:{experiment_name:"",task_type:"",task_desc:"",model_type:"",model_foundation:void 0,recommend:"",files:void 0,tp_tuner:"",tp_max_trials:"",training_params:{}},activeNames:0,rules:{experiment_name:[{required:!0,message:"请输入实验名称",trigger:"blur"},{pattern:/^(?!\s+).*(?<!\s)$/,message:"首尾不能为空格",trigger:"blur"},{pattern:/^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$/,message:"只能输入61位小写英文或数字",trigger:"blur"}],task_type:[{required:!0,message:"请选择任务类型",trigger:"change"}],task_desc:[{required:!0,message:"请输入任务描述",trigger:"blur"}],model_type:[{required:!0,message:"请输入基础模型",trigger:"blur"}],files:[{required:!1,message:"请上传训练数据",trigger:"change"}],tp_tuner:[{required:!0,message:"请选择调优算法",trigger:"change"}],tp_max_trials:[{required:!0,message:"请输入最大试验次数",trigger:"blur"}],training_params:[{required:!0,message:"请输入训练参数",trigger:"blur"}],model_foundation:[{required:!0,message:"请输入基础模型",trigger:"blur"}]},taskList:[],tp_tunerList:[{value:1,label:"greedy"},{value:2,label:"bayesian"},{value:3,label:"hyperband"},{value:4,label:"random"}]}},created:function(){this.init()},mounted:function(){},methods:{init:function(){var e=this;Object(n["j"])().then((function(t){console.log("🚀 ~ file: experimentForm.vue ~ line 24 ~ taskTypes ~ res",t),t.length>0&&(e.taskList=[],t.forEach((function(t){e.taskList.push({value:t,label:t})})))}))},upload:function(){this.form.task_type?["structured-data-classification","structured-data-regression"].includes(this.form.task_type)?(console.log("上传单文件"),document.getElementById("upload-file").click()):(console.log("上传图片文件夹"),document.getElementById("upload-folder").click()):this.$message({type:"warning",message:"请先选择任务类型"})},onFileChanged:function(e){console.log("🚀 ~ onFileChanged ~ e:",e.target,e.target.files),this.uploadNum=e.target.files.length,this.uploadParameters=!0},onFolderChanged:function(e){console.log("🚀 ~ onFolderChanged ~ e:",e.target.files),this.uploadNum=e.target.files.length,this.uploadParameters=!1},handleUploadClose:function(){this.uploadNum=null},changeModelType:function(e,t){console.log("🚀 ~ changeModelType ~ file, fileList:",e,t)},suggest:function(){var e=this;if(!this.form.task_type||!this.form.task_desc)return this.$message({message:"请先选择任务类型和填写任务描述",type:"warning",duration:1500}),!1;this.loading=this.$loading({lock:!0,text:"数据加载中...",spinner:"el-icon-loading",background:"rgba(0, 0, 0, 0.7)"});var t={task_type:this.form.task_type,task_desc:this.form.task_desc,model_nums:1};Object(n["a"])(t).then((function(t){console.log(t,"推荐数据"),e.form.model_foundation=t.candidate_models[0].model_type,e.form.recommend=t.candidate_models[0].reason,e.form.training_params=t.candidate_models[0].training_params,e.dialogVisible=!0})).finally((function(){e.loading.close()}))},recommend:function(){var e=this;this.$refs.form.validateField("model_foundation",(function(t){t||(e.form.model_type=e.form.model_foundation,e.dialogVisible=!1)}))},handleClose:function(e){var t=this;this.$confirm("确定取消吗？").then((function(e){t.dialogVisible=!1})).catch((function(e){}))},handleChange:function(e){console.log(e)}}},m=r,c=(a("32e9"),a("2877")),p=Object(c["a"])(m,s,i,!1,null,null,null),d=p.exports,u=(a("2f62"),{components:{ExperimentForm:d},data:function(){return{dialogFormVisible:!1,tasks:[],taskList:[{value:1,label:"structured-data-classification"},{value:2,label:"structured-data-regression"},{value:3,label:"image-classification"},{value:4,label:"image-regression"}],algorithmList:[{value:1,label:"greedy"},{value:2,label:"bayesian"},{value:3,label:"hyperband"},{value:4,label:"random"}]}},created:function(){this.init()},computed:{},methods:{init:function(){this.getExperimentList()},getExperimentList:function(){var e=this;Object(n["d"])().then((function(t){e.tasks=t.experiment_cards,console.log(t,"实验列表")}))},openDialog:function(){this.dialogFormVisible=!0},info:function(e){this.$router.push({path:"/experiment/detail",query:{experiment_name:e.experiment_name,task_type:e.task_type}})},del:function(e){var t=this;console.log(e,"删除实验"),this.$confirm("此操作将永久删除该文件, 是否继续?","提示",{confirmButtonText:"确定",cancelButtonText:"取消",type:"warning"}).then((function(){console.log(e.experiment_name,"删除实验1"),Object(n["f"])(e.experiment_name).then((function(e){t.$message({type:"success",message:"删除成功!"}),t.getExperimentList()}))})).catch((function(){console.log("已取消删除"),t.$message({type:"info",message:"已取消删除"})}))},openExperiment:function(){var e=this;console.log("🚀 ~ this.$refs.experimentForm.$refs.form.validate ~ this.$refs.experimentForm.form:",this.$refs.experimentForm.form),this.$refs.experimentForm.$refs.form.validate((function(t){if(!t)return!1;var a=JSON.parse(JSON.stringify(e.$refs.experimentForm.form));delete a.recommend,delete a.model_foundation,a.training_params=JSON.stringify(a.training_params),a.tp_tuner=e.algorithmList[a.tp_tuner-1].label;var l=a,o=new FormData;for(var s in l)o.append(s,l[s]);var i=e.$refs.experimentForm.uploadParameters?document.getElementById("upload-file"):document.getElementById("upload-folder");i.files.forEach((function(e){o.append("files",e)})),console.log("🚀 ~ this.$refs.experimentForm.$refs.form.validate ~ params:",l),console.log("🚀 ~ this.$refs.experimentForm.$refs.form.validate ~ p:",o),Object(n["e"])(o).then((function(t){console.log(t,"推荐数据"),e.dialogFormVisible=!1,e.getExperimentList()})).finally((function(){console.log(6666)})),console.log(e.$refs.experimentForm.form,"创建实验参数")}))}}}),f=u,g=(a("e72d"),Object(c["a"])(f,l,o,!1,null,null,null));t["default"]=g.exports},e72d:function(e,t,a){"use strict";a("2023")}}]);
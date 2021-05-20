<template>
  <div style="padding-top: 30px">
    <h2><small>知识抽取</small></h2>
    <el-tabs type="card" v-model="activeName" v-loading="loading" @tab-click="handleClick">
      <el-tab-pane label="Text" name="first">
        <text-area ref="textarea"> </text-area>
            <el-button
      type="primary"
      style="margin: 20px 60px 0px 0px; float: right"
      @click="submit"
      >提交</el-button>
      </el-tab-pane>
      <el-tab-pane label="PMID" name="second">
        <pmid-area ref="pmidarea"> </pmid-area>
            <el-button
      type="primary"
      style="margin: 20px 60px 0px 0px; float: right"
      @click="submit"
      >提交</el-button>
      </el-tab-pane>

      <el-tab-pane label="知识图谱可视化" name="third" >
                <div class="result-legend">
        <div class="legend-entity"> <span class="legend-disease"> </span> disease </div>
        <div class="legend-entity"> <span class="legend-gene" /> gene </div>
         <div class="legend-entity"> <span class="legend-tissue"> </span> tissue </div>
        </div>
      <graph v-if="KGData != null" keys="KG" :data="KGData"> </graph>
      
      </el-tab-pane>
            <el-tab-pane label="知识检索" name="fourth" >
              <search></search>
                      <!-- <graph v-if="KGData != null" keys="KG" :data="KGData"> </graph> -->
      </el-tab-pane>

    </el-tabs>

    <result :resultData="resultData"/>
  </div>
</template>
<script>
import TextArea from "./TextArea.vue";
import PmidArea from "./PmidArea.vue";
import Result from "./Result.vue"
import Graph from "./Graph.vue"
import Search from "./Search.vue"
export default {
  components: {
    TextArea,
    PmidArea,
    Result,
    Graph,
    Search
  },
  data() {
    return {
      activeName: "first",
      content: "",
      contentType: "text",
      resultData: null,
      loading: false,
      KGData: null, 
 test: [
        {
            "sents": [
                {
                    "sent": "Germline mutation in serine / threonine kinase 11 ( STK11 , also called LKB1 ) results in Peutz - Jeghers syndrome , characterized by intestinal hamartomas and increased incidence of epithelial cancers .",
                    "relation": [
                      {
                        "source": 3,
                        "target": 0,
                        "relationship": "gene_mapped_to_disease" ,
                      },
                      {
                        "source": 0, 
                        "target": 3,
                        "relationship": "disease_mapped_to_gene",
                      },
                      {
                        "source": 6, 
                        "target": 0,
                        "relationship": "disease_associated_with_tissue_or_organ",
                      }
                    ],
                    "entity": [
                        {
                            "pos": [
                                17,
                                21
                            ],
                            "name": 
                                "Peutz-Jeghers syndrome",
                            "id": 0,
                            "type": "disease"
                        },
                        {
                            "pos": [
                                30,
                                32
                            ],
                            "name":
                                "epithelial cancers",
                            "id": 1,
                            "type": "disease"
                        },
                        {
                            "pos": [
                                3,
                                8
                            ],
                            "name": 
                                "serine/threonine kinase11",
                            
                            "id": 2,
                            "type": "gene"
                        },
                        {
                            "pos": [
                                9,
                                10
                            ],
                            "name": [
                                "STK11"
                            ],
                            "id": 3,
                            "type": "gene"
                        },
                        {
                            "pos": [
                                13,
                                14
                            ],
                            "name": [
                                "LKB1"
                            ],
                            "id": 4,
                            "type": "gene"
                        },
                        {
                          'name': [
                              'intestinal'
                            ],
                          'pos': [
                              24, 
                              25
                            ],
                            'type': 'tissue',
                            'id': 6
                        }
                    ]
                },
            ]
        }
    ],
    };
  },
  // mounted() {
  //     console.log("test"),
  // },
  methods: {
    async handleClick() {

      this.KGData = null
        if(this.activeName == 'third') {
              this.resultData = null
              this.loading = true
              const res = await this.axios.get("http://localhost:8081/kg/get");
             if(res.status == 200 && res.data.success == true) {
                this.KGData = {
                  entity: res.data.data.nodes,
                  relation: res.data.data.relations
                };
                console.log(this.KGData);
                this.$message({
                    type: 'success',
                    message: res.data.message,
                });
            } else{
                    this.$message({
                    type: 'error',
                    message: res.status==200?res.data.message:"未知错误",
                });
            }
            this.loading = false
        }
    },
    async submit() {
      this.resultData = null
      this.loading = true
      if (this.activeName === "first") {
        this.contentType = "text"
        if (this.$refs.textarea.text === "") {
          this.content = this.$refs.textarea.defaultText;
          this.$refs.textarea.text = this.$refs.textarea.defaultText;
        } else {
          this.content = this.$refs.textarea.text;
        }
      } else {
        this.contentType="PMIDS"
        if (this.$refs.pmidarea.PMIDs === "") {
          this.content = this.$refs.pmidarea.defaultPMIDs;
          this.$refs.pmidarea.PMIDs = this.$refs.pmidarea.defaultPMIDs;
        } else {
          this.content = this.$refs.pmidarea.PMIDs;
        }
      }
      const res = await this.axios.post("http://localhost:8081/model/submit",
        {
          contentType: this.contentType,
          content: this.content,
        },
      );
       if(res.status == 200 && res.data.success == true) {
                this.resultData = res.data.data
                this.$message({
                    type: 'success',
                    message: res.data.message,
                });
            } else{
                    this.$message({
                    type: 'error',
                    message: res.status==200?res.data.message:"未知错误",
                });
        }
      
      this.loading = false
    },
  },
};
</script>

<style>
.result-legend {
    /* width: 60%; */
    display: flex;
    flex: 1;
    min-height: 1px;
}
.legend-entity {
    padding-right: 15px;
    padding-left: 15px;
    align-items: center;
    text-align: inherit;
}
.legend-disease::before {
    display: inline-block;
    width:16px;
    height:16px;
    content: "";
    border-radius:100px;
    background: rgba(255, 0, 0, 0.3);
}
.legend-gene::before {
    display: inline-block;
    width:16px;
    height:16px;
    content: "";
    align-content:inherit;
    border-radius:100px;
    background: rgba(0, 0, 255, 0.3);
}
.legend-tissue::before {
    display: inline-block;
    width:16px;
    height:16px;
    content: "";
    align-content:inherit;
    border-radius:100px;
    background: rgba(255, 225, 0, 0.6);
}
  .el-select .el-input {
    width: 130px;
  }
  .input-with-select .el-input-group__prepend {
    background-color: #fff;
  }
</style>

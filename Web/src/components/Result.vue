<template>
<!-- 29284222 -->
    <div v-if="resultData!=null" style="margin: 60px 60px 0 30px">
        <div class="result-header">
        <div style="color: #777;font-size: 18px; width: 70%;">结果</div>
        <div class="result-legend">
        <div class="legend-entity"> <span class="legend-disease"> </span> disease </div>
        <div class="legend-entity"> <span class="legend-gene" /> gene </div>
         <div class="legend-entity"> <span class="legend-tissue"> </span> tissue </div>
        </div>
        </div>
        <div v-for="(item, index) in resultData" :key="index">
        <p style="color777: #" v-if="hasPmid(item)"> PMID: {{item.PMID}}</p>
        <el-collapse v-for="(i, sent_id) in item.sents" :key="sent_id">

          <el-collapse-item name="1">
              <template v-slot:title >
                <div style="overflow:hidden">
              <template v-for="(word, word_id) in i.sent"  :key="word_id" style="width: 100%;">
                  <span
                  style="white-space:pre; font-size:14px"
                  :class="{'entity': i.sent_type[word_id] > 0, 
                        'gene': (i.sent_type[word_id] >> geneTypeId) == 1, 
                        'disease': (i.sent_type[word_id] >> diseaseTypeId) == 1,
                        'tissue':  (i.sent_type[word_id] >> tissueTypeId) == 1,}"
                  >
                    {{process(i.sent, word, word_id)}}
                </span>
              </template>
              </div>
          </template>
          <div class='align'>
            <div style="width:90%; align-items: center; padding-right: 70px">{{i.entity}}</div>
              <div>
              <el-button
      type="success"
      @click="saveToKG(i)"
      style="float:right;"
      >保存</el-button>
      </div>
      </div>
            <graph :keys="getKeys(index, sent_id)" :data="{entity: i.entity, relation: i.relation}"> </graph>

          </el-collapse-item>
        </el-collapse>
           </div>
      </div>
</template>

<script>
import Graph from "./Graph.vue";
export default {
    components:{
        Graph
    },
    props:{
        resultData: {
            type: Object
        }
    },
    data() {
        return {
            activeName: '1', 
            diseaseTypeId: 0,
            geneTypeId: 1,
            tissueTypeId: 2,
            articles: null,
        }
    }, 
     methods: {
         process(sent, word, word_id){
            let word_list = sent
            if(word.length == 1 || word_list[word_id+1].length == 1) {
                return word;
            } else return word.concat(" ");
            
         },
         isEntity(entity, word_id) {
             this.isDisease = false
             this.isGene = false
             this.isTissue = false
             
            var flag = false
            // console.log(entity)
             for(const item of entity) {
                 if(word_id >= item.pos[0] && word_id < item.pos[1]) {
                     if(item.type == "disease") this.isDisease = true
                     else if(item.type == "gene") this.isGene = true
                     else if(item.type == "tissue") this.isTissue = true
                    flag = true
                 }
             }
             return flag
         },
        hasPmid(item) {
            if("PMID" in item) return true
            else return false
        },
        getKeys(index, sent_id) {
            return ((index + 1) * 1000 + sent_id).toString()
        },
        async saveToKG(data) {
            let nodes = data.entity
            let relations = []
            for(const item of data.relation) {
				var sourceNode = nodes.filter(function (n) {
	                    return n.id === item.source;
	                })[0];
	            if (typeof(sourceNode) == 'undefined') continue;
	            var targetNode = nodes.filter(function (n) {
	                    return n.id === item.target;
	                })[0];
	            if (typeof(targetNode) == 'undefined') continue;

				this.relations.push({
					source: sourceNode.name,
					target: targetNode.name,
					relationship: item.relationship,
				})
			}
            const res = await this.axios.post("http://localhost:8081/kg/save",
                {
                    nodes: nodes,
                    relations: relations,
                },
            );
            if(res.status == 200 && res.data.success == true) {
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
        }
    }
}
</script>

<style>
.entity {
    text-decoration: none;
    /* background: rgba(255, 0, 0, 0.3); */
    line-height: 16px;
}
.disease {
    background: rgba(255, 0, 0, 0.3);
}
.gene {
    background: rgba(0, 0, 255, 0.3);
}
.tissue{
    background: rgba(255,255,0.6);
}
.align {
    display: flex;
    flex: 1;
}
.result-header {
    display: flex;
    flex: 1;
}
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

</style>
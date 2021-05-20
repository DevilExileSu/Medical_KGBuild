<template>
    <div style="margin: 0px 60px 0 30px">
    <p style="color: #777">输入以“|”分隔的同一类别关键词</p>
   <el-input
    placeholder=""
    v-model="keyWord"
    class="input-with-select"
    v-loading="loading"
  >
    <template #prepend>
      <el-select v-model="type" placeholder="请选择">
        <el-option label="gene" value="gene"></el-option>
        <el-option label="disease" value="disease"></el-option>
        <el-option label="tissue" value="tissue"></el-option>
      </el-select>
    </template>
    <template #append>
      <el-button icon="el-icon-search" @click="search"></el-button>
    </template>
  </el-input>
    <div v-if="resultData != null">
        <div class="result-legend" style="padding-top:20px">
        <div class="legend-entity"> <span class="legend-disease"> </span> disease </div>
        <div class="legend-entity"> <span class="legend-gene" /> gene </div>
         <div class="legend-entity"> <span class="legend-tissue"> </span> tissue </div>
        </div>
      <graph  keys="result" :data="resultData" :keyWord="kw"> </graph>
    </div>
</div>

</template>

<script>
// import {ref } from 'vue'
import Graph from "./Graph.vue";
export default {
      components: {
    Graph
  },
  data() {
    return {
        keyWord: "",
        type: "",
        resultData: null,
        loading: false,
        kw:[],
    };
  },
  methods: {
      async search(){
          this.loading = true
          this.resultData=null
        const res = await this.axios.post("http://localhost:8081/kg/find/",
        {
          type: this.type,
          keyWord: this.keyWord,
        },
      );
       if(res.status == 200 && res.data.success == true) {
                this.resultData = {
                  entity: res.data.data.nodes,
                  relation: res.data.data.relations
                };
                this.kw = this.keyWord.split('|').map(function(i) {return i.trim()})
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
    
  }
}
</script>

<style>

</style>
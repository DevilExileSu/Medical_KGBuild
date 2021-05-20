<template>
  <el-header>
    <h3>基于深度学习的医学文献知识图谱构建</h3>
  </el-header>
  <knowledge-extract> </knowledge-extract>
</template>

<script>
import KnowledgeExtract from "./components/KnowledgeExtract.vue";
export default {
  name: "App",
  components: {
    KnowledgeExtract,
  },
};
</script>

<style>
#app {
  font-family: "Arial", "Microsoft YaHei", "黑体", "宋体", sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.el-header {
  background-color: #f8f8f8;
  color: #333;
  text-align: center;
  line-height: 60px;
  box-shadow: 2px 2px 2px 2px #f0f0f0;
}
</style>

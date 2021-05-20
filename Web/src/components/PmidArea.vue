<template>
  <div style="margin: 0px 60px 0 30px">
    <p style="color: #777">输入以逗号分隔的PMID</p>
    <el-input v-model="PMIDs" :placeholder="defaultPMIDs"> </el-input>
  </div>
</template>

<script>
export default {
  data() {
    return {
      defaultPMIDs: "17676035,32949396",
      PMIDs: "",
    };
  },
};
</script>

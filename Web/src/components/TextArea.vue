<template>
  <div style="margin: 0px 60px 0 30px">
    <p style="color: #777">输入任何生物医学文本 (文本最大长度为1024)</p>
    <el-input
      type="textarea"
      :autosize="{ minRows: 15, maxRows: 25 }"
      :placeholder="defaultText"
      v-model="text"
    >
    </el-input>
  </div>
</template>

<script>
export default {
  data() {
    return {
      defaultText:
        "Germline mutation in serine/threonine kinase 11 (STK11, also called LKB1) results in Peutz-Jeghers syndrome, characterized by intestinal hamartomas and increased incidence of epithelial cancers. Although uncommon in most sporadic cancers, inactivating somatic mutations of LKB1 have been reported in primary human lung adenocarcinomas and derivative cell lines. Here we used a somatically activatable mutant Kras-driven model of mouse lung cancer to compare the role of Lkb1 to other tumour suppressors in lung cancer. Although Kras mutation cooperated with loss of p53 or Ink4a/Arf (also known as Cdkn2a) in this system, the strongest cooperation was seen with homozygous inactivation of Lkb1. Lkb1-deficient tumours demonstrated shorter latency, an expanded histological spectrum (adeno-, squamous and large-cell carcinoma) and more frequent metastasis compared to tumours lacking p53 or Ink4a/Arf. Pulmonary tumorigenesis was also accelerated by hemizygous inactivation of Lkb1. Consistent with these findings, inactivation of LKB1 was found in 34% and 19% of 144 analysed human lung adenocarcinomas and squamous cell carcinomas, respectively. Expression profiling in human lung cancer cell lines and mouse lung tumours identified a variety of metastasis-promoting genes, such as NEDD9, VEGFC and CD24, as targets of LKB1 repression in lung cancer. These studies establish LKB1 as a critical barrier to pulmonary tumorigenesis, controlling initiation, differentiation and metastasis. ",
      text: "",
    };
  },
};
</script>

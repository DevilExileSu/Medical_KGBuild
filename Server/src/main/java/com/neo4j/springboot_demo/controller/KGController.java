package com.neo4j.springboot_demo.controller;


import com.neo4j.springboot_demo.entity.nodes.DiseaseNode;
import com.neo4j.springboot_demo.entity.nodes.GeneNode;
import com.neo4j.springboot_demo.entity.nodes.TissueNode;
import com.neo4j.springboot_demo.response.Result;
import com.neo4j.springboot_demo.response.ResultCode;
import com.neo4j.springboot_demo.service.DiseaseNodeService;
import com.neo4j.springboot_demo.service.GeneNodeService;
import com.neo4j.springboot_demo.service.TissueNodeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@Controller
@ResponseBody
@RequestMapping(value="/kg")
@CrossOrigin
public class KGController {
    @Autowired
    private TissueNodeService tissueNodeService;

    @Autowired
    private DiseaseNodeService diseaseNodeService;

    @Autowired
    private GeneNodeService geneNodeService;



    // 四大关系： gene->disease, disease->tissue, disease->disease, tissue->tissue
    @PostMapping("/save")
    Result saveNodesAndRelationships(@RequestBody Map<String, Object> map){
        Optional<List<Map>> nodesOptinal = Optional.ofNullable((List<Map>) map.get("nodes"));
        Optional<List<Map>> relationsOptinal = Optional.ofNullable((List<Map>) map.get("relations"));
        StringBuilder msgBuilder = new StringBuilder();
        if(nodesOptinal.isPresent()) {
            List<Map> nodes = nodesOptinal.get();
            for (Map node : nodes) {
                String type = (String) node.get("type");
                String name = (String) node.get("name");
                switch (type) {
                    case "disease":
                        Optional<DiseaseNode> diseaseNodeOptional = diseaseNodeService.getDiseaseNodeByDiseaseName(name);
                        if (!diseaseNodeOptional.isPresent()) {
                            diseaseNodeService.saveDiseaseNodeByDiseaseName(name);
                        }
                        break;
                    case "gene":
                        Optional<GeneNode> geneNodeOptional = geneNodeService.getGeneNodeByGeneName(name);
                        if (!geneNodeOptional.isPresent()) {
                            geneNodeService.saveGeneNodeByGeneName(name);
                        }
                        break;

                    case "tissue":
                        Optional<TissueNode> tissueNodeOptional = tissueNodeService.getTissueNodeByTissueName(name);
                        if (!tissueNodeOptional.isPresent()) {
                            tissueNodeService.saveTissueNodeByTissueName(name);
                        }
                        break;
                    default:
                        msgBuilder.append("未知节点类型: ")
                                .append(type)
                                .append("，节点: ")
                                .append(name)
                                .append("保存失败. \n");
                }
            }
        }
        if(relationsOptinal.isPresent()) {
            List<Map> relations = relationsOptinal.get();
            for(Map relation : relations) {
                String relationship = (String) relation.get("relationship");
                String sourceName = (String) relation.get("source");
                String targetName = (String) relation.get("target");
                switch(relationship) {
                    case "gene_associated_with_disease":
                        diseaseNodeService.saveRelationWithGeneNode(targetName, sourceName);
                        break;
                    case "disease_associated_with_tissue":
                        diseaseNodeService.saveRelationWithTissueNode(sourceName, targetName);
                        break;
                    case "disease_associated_with_disease":
                        diseaseNodeService.saveRelationWithDiseaseNode(sourceName, targetName);
                        break;
                    case "tissue_associated_with_tissue":
                        tissueNodeService.saveRelationWithTissueNode(sourceName, targetName);
                        break;
                    default:
                        msgBuilder.append("未知关系类型: ")
                                .append(relationship)
                                .append("，源节点: ")
                                .append(sourceName)
                                .append(", 目标节点: ")
                                .append(targetName)
                                .append("保存失败. \n");

                }
            }
        }
        if(msgBuilder.length() == 0) {
            return Result.SUCCESS();
        }else {
            return new Result(20002, msgBuilder.toString(), false);
        }
    }

    /*
    resNodes = [
        {
            name: ,
            id: ,
            type: ,
        }
    ]
    resRelations = [
        {
            sourceId: ,
            targetId: ,
            relationship
        }
    ]
    * */
    @GetMapping("/get")
    Result getNodesAndRelationships() {
        List<DiseaseNode> diseaseNodeList = diseaseNodeService.listDiseaseNodes();
        List<GeneNode> geneNodeList = geneNodeService.listGeneNodes();
        List<TissueNode> tissueNodeList = tissueNodeService.listTissueNodes();
        List<Map> resNodes = new ArrayList<>();
        List<Map> resRelations = new ArrayList<>();
        Map<String, Object> resData = new HashMap<>();
        Set<Long> diseaseNodeSet = new HashSet<>();
        Set<Long> tissueNodeSet = new HashSet<>();
        Set<Long> geneNodeSet = new HashSet<>();

        for(DiseaseNode diseaseNode : diseaseNodeList) {
            String diseaseName = diseaseNode.getDiseaseName();
            Long sourceId = diseaseNode.getId();
            // 疾病节点数据
            diseaseNodeSet.add(sourceId);
            Optional<Map<String, Object>> tmp = Optional.ofNullable(getRelationsByNodeName("disease", diseaseName, 1));
            if(tmp.isPresent()) {
                Map<String, Object> res = tmp.get();
                resRelations.addAll((List<Map>) res.get("relations"));
                diseaseNodeSet.addAll((Set<Long>) res.get("diseaseNodeset"));
                tissueNodeSet.addAll((Set<Long>) res.get("tissueNodeSet"));
                geneNodeSet.addAll((Set<Long>) res.get("geneNodeSet"));
            }
        }

        for(GeneNode geneNode : geneNodeList) {
            String geneName = geneNode.getGeneName();
            Long sourceId = geneNode.getId();
            // 基因节点数据
            geneNodeSet.add(sourceId);
            Optional<Map<String, Object>> tmp = Optional.ofNullable(getRelationsByNodeName("gene", geneName, 1));
            if(tmp.isPresent()) {
                Map<String, Object> res = tmp.get();
                resRelations.addAll((List<Map>) res.get("relations"));
                diseaseNodeSet.addAll((Set<Long>) res.get("diseaseNodeset"));
                tissueNodeSet.addAll((Set<Long>) res.get("tissueNodeSet"));
                geneNodeSet.addAll((Set<Long>) res.get("geneNodeSet"));
            }
        }

        for(TissueNode tissueNode : tissueNodeList) {
            String tissueName = tissueNode.getTissueName();
            Long tissueId = tissueNode.getId();
            tissueNodeSet.add(tissueId);
            Optional<Map<String, Object>> tmp = Optional.ofNullable(getRelationsByNodeName("tissue", tissueName, 1));
            if(tmp.isPresent()) {
                Map<String, Object> res = tmp.get();
                resRelations.addAll((List<Map>) res.get("relations"));
                diseaseNodeSet.addAll((Set<Long>) res.get("diseaseNodeset"));
                tissueNodeSet.addAll((Set<Long>) res.get("tissueNodeSet"));
                geneNodeSet.addAll((Set<Long>) res.get("geneNodeSet"));
            }
        }

        resNodes.addAll(getNodesFromSets("disease", diseaseNodeSet));
        resNodes.addAll(getNodesFromSets("gene", geneNodeSet));
        resNodes.addAll(getNodesFromSets("tissue", tissueNodeSet));

        resData.put("nodes", resNodes);
        resData.put("relations", resRelations);
        return new Result(ResultCode.SUCCESS, resData);
    }

    Map<String, Object> getRelationsByNodeName(String type, String name, int k) {
        Set<Long> diseaseNodeSet = new HashSet<>();
        Set<Long> tissueNodeSet = new HashSet<>();
        Set<Long> geneNodeSet = new HashSet<>();
        List<Map> resRelations = new ArrayList<>();
        Map<String, Object> res = new HashMap<>();
        if(k == 0) {
            return null;
        }
        switch (type) {
            case "disease":
                Optional<DiseaseNode> diseaseNodeOptional = diseaseNodeService.getDiseaseNodeByDiseaseName(name);
                if (diseaseNodeOptional.isPresent()) {
                    DiseaseNode diseaseNode = diseaseNodeOptional.get();
                    Long diseaseId = diseaseNode.getId();
                    diseaseNodeSet.add(diseaseId);
                    List<DiseaseNode> targetDiseaseNodes = diseaseNodeService.getRelationWithDiseaseNode(name);
                    for(DiseaseNode targetDiseaseNode : targetDiseaseNodes) {
                        Map<String, Object> relations = new HashMap<>();
                        relations.put("source", diseaseId);
                        relations.put("target", targetDiseaseNode.getId());
                        relations.put("relationship", "disease_associated_with_disease");
                        diseaseNodeSet.add(targetDiseaseNode.getId());
                        resRelations.add(relations);
                        Optional<Map<String, Object>> tmp = Optional
                                .ofNullable(getRelationsByNodeName("disease", targetDiseaseNode.getDiseaseName(), k--));
                        if(tmp.isPresent()) {
                            Map<String, Object> nebRes = tmp.get();
                            resRelations.addAll((List<Map>) nebRes.get("relations"));
                            diseaseNodeSet.addAll((Set<Long>) nebRes.get("diseaseNodeset"));
                            tissueNodeSet.addAll((Set<Long>) nebRes.get("tissueNodeSet"));
                            geneNodeSet.addAll((Set<Long>) nebRes.get("geneNodeSet"));
                        }
                    }

                    List<GeneNode> sourceGeneNodes = geneNodeService.getRelationWithDiseaseNode(name);
                    for(GeneNode sourceGeneNode : sourceGeneNodes) {
                        Map<String, Object> relations = new HashMap<>();
                        relations.put("source", sourceGeneNode.getId());
                        relations.put("target", diseaseId);
                        relations.put("relationship", "gene_associated_with_disease");
                        geneNodeSet.add(sourceGeneNode.getId());
                        resRelations.add(relations);
                        Optional<Map<String, Object>> tmp = Optional
                                .ofNullable(getRelationsByNodeName("disease", sourceGeneNode.getGeneName(), k--));
                        if(tmp.isPresent()) {
                            Map<String, Object> nebRes = tmp.get();
                            resRelations.addAll((List<Map>) nebRes.get("relations"));
                            diseaseNodeSet.addAll((Set<Long>) nebRes.get("diseaseNodeset"));
                            tissueNodeSet.addAll((Set<Long>) nebRes.get("tissueNodeSet"));
                            geneNodeSet.addAll((Set<Long>) nebRes.get("geneNodeSet"));
                        }
                    }

                    List<TissueNode> targetTissueNodes = tissueNodeService.getRelationWithDiseaseNode(name);
                    for(TissueNode targetTissueNode : targetTissueNodes) {
                        Map<String, Object> relations = new HashMap<>();
                        relations.put("source", diseaseId);
                        relations.put("target", targetTissueNode.getId());
                        relations.put("relationship", "disease_associated_with_tissue");
                        tissueNodeSet.add(targetTissueNode.getId());
                        resRelations.add(relations);
                        Optional<Map<String, Object>> tmp = Optional
                                .ofNullable(getRelationsByNodeName("disease", targetTissueNode.getTissueName(), k--));
                        if(tmp.isPresent()) {
                            Map<String, Object> nebRes = tmp.get();
                            resRelations.addAll((List<Map>) nebRes.get("relations"));
                            diseaseNodeSet.addAll((Set<Long>) nebRes.get("diseaseNodeset"));
                            tissueNodeSet.addAll((Set<Long>) nebRes.get("tissueNodeSet"));
                            geneNodeSet.addAll((Set<Long>) nebRes.get("geneNodeSet"));
                        }
                    }

                }
                break;
            case "gene":
                Optional<GeneNode> geneNodeOptional = geneNodeService.getGeneNodeByGeneName(name);
                if (geneNodeOptional.isPresent()) {
                    GeneNode sourceNode = geneNodeOptional.get();
                    Long sourceId = sourceNode.getId();
                    geneNodeSet.add(sourceId);
                    List<DiseaseNode> targetDiseaseNodes = diseaseNodeService.getRelationWithGeneNode(name);
                    for(DiseaseNode targetDiseaseNode : targetDiseaseNodes) {
                        Map<String, Object> relations = new HashMap<>();
                        relations.put("source", sourceId);
                        relations.put("target", targetDiseaseNode.getId());
                        relations.put("relationship", "gene_associated_with_disease");
                        diseaseNodeSet.add(targetDiseaseNode.getId());
                        resRelations.add(relations);
                        Optional<Map<String, Object>> tmp = Optional
                                .ofNullable(getRelationsByNodeName("disease", targetDiseaseNode.getDiseaseName(), k--));
                        if(tmp.isPresent()) {
                            Map<String, Object> nebRes = tmp.get();
                            resRelations.addAll((List<Map>) nebRes.get("relations"));
                            diseaseNodeSet.addAll((Set<Long>) nebRes.get("diseaseNodeset"));
                            tissueNodeSet.addAll((Set<Long>) nebRes.get("tissueNodeSet"));
                            geneNodeSet.addAll((Set<Long>) nebRes.get("geneNodeSet"));
                        }
                    }
                }
                break;
            case "tissue":
                Optional<TissueNode> tissueNodeOptional = tissueNodeService.getTissueNodeByTissueName(name);
                if(tissueNodeOptional.isPresent()) {
                    TissueNode tissueNode = tissueNodeOptional.get();
                    Long tissueId = tissueNode.getId();
                    tissueNodeSet.add(tissueId);
                    List<TissueNode> targetTissueNodes = tissueNodeService.getRelationWithTissueNode(name);
                    for(TissueNode targetTissueNode:targetTissueNodes) {
                        Map<String, Object> relations = new HashMap<>();
                        relations.put("source", tissueId);
                        relations.put("target", targetTissueNode.getId());
                        relations.put("relationship", "gene_associated_with_disease");
                        tissueNodeSet.add(targetTissueNode.getId());
                        resRelations.add(relations);
                        Optional<Map<String, Object>> tmp = Optional
                                .ofNullable(getRelationsByNodeName("disease", targetTissueNode.getTissueName(), k--));
                        if(tmp.isPresent()) {
                            Map<String, Object> nebRes = tmp.get();
                            resRelations.addAll((List<Map>) nebRes.get("relations"));
                            diseaseNodeSet.addAll((Set<Long>) nebRes.get("diseaseNodeset"));
                            tissueNodeSet.addAll((Set<Long>) nebRes.get("tissueNodeSet"));
                            geneNodeSet.addAll((Set<Long>) nebRes.get("geneNodeSet"));
                        }
                    }
                    List<DiseaseNode> sourceDiseaseNodes = diseaseNodeService.getRelationWithTissueNode(name);
                    for(DiseaseNode sourceDiseaseNode : sourceDiseaseNodes) {
                        Map<String, Object> relations = new HashMap<>();
                        relations.put("source", sourceDiseaseNode.getId());
                        relations.put("target", tissueId);
                        relations.put("relationship", "disease_associated_with_tissue");
                        diseaseNodeSet.add(sourceDiseaseNode.getId());
                        resRelations.add(relations);
                        Optional<Map<String, Object>> tmp = Optional
                                .ofNullable(getRelationsByNodeName("disease", sourceDiseaseNode.getDiseaseName(), k--));
                        if(tmp.isPresent()) {
                            Map<String, Object> nebRes = tmp.get();
                            resRelations.addAll((List<Map>) nebRes.get("relations"));
                            diseaseNodeSet.addAll((Set<Long>) nebRes.get("diseaseNodeset"));
                            tissueNodeSet.addAll((Set<Long>) nebRes.get("tissueNodeSet"));
                            geneNodeSet.addAll((Set<Long>) nebRes.get("geneNodeSet"));
                        }
                    }
                }
                break;
        }
        res.put("relations", resRelations);
        res.put("diseaseNodeset", diseaseNodeSet);
        res.put("tissueNodeSet", tissueNodeSet);
        res.put("geneNodeSet", geneNodeSet);
        return res;
    }
    List<Map<String, Object>> getNodesFromSets(String type, Set<Long> set) {
        List<Map<String, Object>> resNodes = new ArrayList<>();
        for(Long id : set) {
            Map<String, Object> tmpNode = new HashMap<>();
            switch(type) {
                case "disease":
                    Optional<DiseaseNode> diseaseNodeOptional = diseaseNodeService.getDiseaseNodeById(id);
                    if(diseaseNodeOptional.isPresent()) {
                        DiseaseNode diseaseNode = diseaseNodeOptional.get();
                        tmpNode.put("name", diseaseNode.getDiseaseName());
                        tmpNode.put("id", id);
                        tmpNode.put("type", "disease");
                        resNodes.add(tmpNode);
                    }
                    break;

                case "gene":
                    Optional<GeneNode> geneNodeOptional = geneNodeService.getGeneNodeById(id);
                    if (geneNodeOptional.isPresent()) {
                        GeneNode geneNode = geneNodeOptional.get();
                        tmpNode.put("name", geneNode.getGeneName());
                        tmpNode.put("id", id);
                        tmpNode.put("type", "gene");
                        resNodes.add(tmpNode);
                    }
                    break;
                case "tissue":
                    Optional<TissueNode> tissueNodeOptional = tissueNodeService.getTissueNodeById(id);
                    if (tissueNodeOptional.isPresent()) {
                        TissueNode tissueNode = tissueNodeOptional.get();
                        tmpNode.put("name", tissueNode.getTissueName());
                        tmpNode.put("id", id);
                        tmpNode.put("type", "tissue");
                        resNodes.add(tmpNode);
                    }
                    break;
            }
        }
        return resNodes;
    }

    @PostMapping("/find")
    Result findNodesAndRelationByNodeName(@RequestBody Map<String, Object> map){
        try {
            String type = (String) map.get("type");
            System.out.println(type);
            List<String> nodeNames = Arrays.asList(((String) map.get("keyWord")).split("\\|"));
            List<Map> resRelations = new ArrayList<>();
            List<Map> resNodes = new ArrayList<>();
            Map<String, Object> resData = new HashMap<>();
            Set<Long> diseaseNodeSet = new HashSet<>();
            Set<Long> tissueNodeSet = new HashSet<>();
            Set<Long> geneNodeSet = new HashSet<>();
            for(String name:nodeNames) {
                name = name.trim();
                System.out.println(name);
                System.out.println(name);
                Optional<Map<String, Object>> tmp = Optional.ofNullable(getRelationsByNodeName(type, name, 2));
                if(tmp.isPresent()) {
                    Map<String, Object> res = tmp.get();
                    System.out.println(res);
                    resRelations.addAll((List<Map>) res.get("relations"));
                    diseaseNodeSet.addAll((Set<Long>) res.get("diseaseNodeset"));
                    tissueNodeSet.addAll((Set<Long>) res.get("tissueNodeSet"));
                    geneNodeSet.addAll((Set<Long>) res.get("geneNodeSet"));
                }
            }
            resNodes.addAll(getNodesFromSets("disease", diseaseNodeSet));
            resNodes.addAll(getNodesFromSets("gene", geneNodeSet));
            resNodes.addAll(getNodesFromSets("tissue", tissueNodeSet));

            resData.put("nodes", resNodes);
            resData.put("relations", resRelations);
            return new Result(ResultCode.SUCCESS, resData);
        } catch(Exception e) {
            System.out.println(e);
            return new Result(ResultCode.FAIL);
        }
    }
}

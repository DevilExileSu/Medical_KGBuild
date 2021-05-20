package com.neo4j.springboot_demo;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.neo4j.springboot_demo.entity.nodes.DiseaseNode;
import com.neo4j.springboot_demo.entity.nodes.GeneNode;
import com.neo4j.springboot_demo.entity.nodes.TissueNode;
import com.neo4j.springboot_demo.service.DiseaseNodeService;
import org.jsoup.nodes.Document;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;

@SpringBootTest
class SpringbootDemoApplicationTests {

    @Autowired
    private DiseaseNodeService diseaseNodeService;
    @Test
    void contextLoads() {
        diseaseNodeService.saveRelationWithGeneNode("lung cancer", "CD24");
//        diseaseNodeService.saveRelationWithGeneNode("lung cancer", "Lkb1");
//        diseaseNodeService.saveRelationWithGeneNode("lung cancer", "NEDD9");
        List<DiseaseNode> diseaseNodes = diseaseNodeService.getRelationWithGeneNode("CD24");
        if(diseaseNodes.size() > 0) {
            for (DiseaseNode diseaseNode : diseaseNodes) {
                System.out.println(diseaseNode.getDiseaseName());
            }
        }
    }
}

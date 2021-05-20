package com.neo4j.springboot_demo.controller;

import com.neo4j.springboot_demo.entity.nodes.DiseaseNode;
import com.neo4j.springboot_demo.response.Result;
import com.neo4j.springboot_demo.response.ResultCode;
import com.neo4j.springboot_demo.service.DiseaseNodeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.Optional;

@Controller
@ResponseBody
@RequestMapping(value="/disease")
@CrossOrigin
public class DiseaseNodeController {
//
//    @Autowired
//    private DiseaseNodeService diseaseNodeService;
//
//    @PostMapping("/create")
//    public Result saveDiseaseNodeByDiseaseName(@RequestBody Map<String, Object> map) {
//        String diseaseName = (String) map.get("diseaseName");
//        if(diseaseName == null || "".equals(diseaseName)) {
//            return Result.NOT_EXIST();
//        }else {
//            diseaseNodeService.saveDiseaseNodeByDiseaseName(diseaseName);
//            return Result.SUCCESS();
//        }
//    }
//
//    @GetMapping(value="/{id}")
//    public Result findById(@PathVariable Long id) {
//        Optional<DiseaseNode> diseaseNodeOptional = diseaseNodeService.getDiseaseNodeById(id);
//        if(diseaseNodeOptional.isPresent()) {
//            return new Result(ResultCode.SUCCESS, diseaseNodeOptional.get());
//        }else {
//            return Result.NOT_EXIST();
//        }
//    }
//
//    @GetMapping("/all")
//    public Result findAll() {
//        List<DiseaseNode> diseaseNodeList = diseaseNodeService.listDiseaseNodes();
//        return new Result(ResultCode.SUCCESS, diseaseNodeList);
//    }
//
//    @DeleteMapping("/delete/{id}")
//    public Result deleteById(@PathVariable Long id) {
//        diseaseNodeService.deleteById(id);
//        return Result.SUCCESS();
//    }
//
//    @PostMapping("/find")
//    public Result findNyDiseaseName(@RequestBody Map<String, Object> map) {
//        String diseaseName = (String) map.get("diseaseName");
//        if(diseaseName == null || "".equals(diseaseName)) {
//            return Result.NOT_EXIST();
//        }else {
//            Optional<DiseaseNode>  diseaseNodeOptional = diseaseNodeService.getDiseaseNodeByDiseaseName(diseaseName);
//            return diseaseNodeOptional.map(diseaseNode -> new Result(ResultCode.SUCCESS, diseaseNode)).orElseGet(Result::NOT_EXIST);
//        }
//    }
}

package com.neo4j.springboot_demo.controller;

import com.neo4j.springboot_demo.entity.nodes.GeneNode;
import com.neo4j.springboot_demo.response.Result;
import com.neo4j.springboot_demo.response.ResultCode;
import com.neo4j.springboot_demo.service.GeneNodeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.Optional;

@Controller
@ResponseBody
@RequestMapping(value="/gene")
@CrossOrigin
public class GeneNodeController {

//    @Autowired
//    private GeneNodeService geneNodeService;
//
//    @PostMapping("/create")
//    public Result saveGeneNodeByGeneName(@RequestBody Map<String, Object> map) {
//        String geneName = (String) map.get("geneName");
//        if(geneName == null || "".equals(geneName)) {
//            return Result.NOT_EXIST();
//        }else {
//            geneNodeService.saveGeneNodeByGeneName(geneName);
//            return Result.SUCCESS();
//        }
//    }
//
//    @GetMapping(value="/{id}")
//    public Result findById(@PathVariable Long id) {
//        Optional<GeneNode> geneNodeOptional = geneNodeService.getGeneNodeById(id);
//        if(geneNodeOptional.isPresent()) {
//            return new Result(ResultCode.SUCCESS, geneNodeOptional.get());
//        }else {
//            return Result.NOT_EXIST();
//        }
//    }
//
//    @GetMapping("/all")
//    public Result findAll() {
//        List<GeneNode> geneNodeList = geneNodeService.listGeneNodes();
//        return new Result(ResultCode.SUCCESS, geneNodeList);
//    }
//
//    @DeleteMapping("/delete/{id}")
//    public Result deleteById(@PathVariable Long id) {
//        geneNodeService.deleteById(id);
//        return Result.SUCCESS();
//    }
//
//    @PostMapping("/find")
//    public Result findByGeneName(@RequestBody Map<String, Object> map) {
//        String geneName = (String) map.get("geneName");
//        if(geneName == null || "".equals(geneName)) {
//            return Result.NOT_EXIST();
//        }else {
//            Optional<GeneNode>  geneNodeOptional = geneNodeService.getGeneNodeByGeneName(geneName);
//            return geneNodeOptional.map(geneNode -> new Result(ResultCode.SUCCESS, geneNode)).orElseGet(Result::NOT_EXIST);
//        }
//    }
}

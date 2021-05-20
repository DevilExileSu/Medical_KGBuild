package com.neo4j.springboot_demo.controller;


import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.neo4j.springboot_demo.response.Result;
import com.neo4j.springboot_demo.response.ResultCode;
import com.neo4j.springboot_demo.utils.GetAbstractUtil;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Controller
@ResponseBody
@RequestMapping(value="/model")
@CrossOrigin
public class ModelController {

    @PostMapping("/submit")
    Result nerModel(@RequestBody Map<String, Object> map) {
        String contentType = (String) map.get("contentType");
        String content = (String) map.get("content");
        Map<String, Object> requestData = new HashMap<>();
        Map<String, Object> data = new HashMap<>();
        if("text".equals(contentType)) {
            List<String> sents = Arrays.asList(content
                    .replaceAll("([a-z0-9])\\. ([A-Z])", "$1\\.\n$2")
                    .split("\n"));
            data.put("sents", sents);
            requestData.put("contentType", contentType);
            requestData.put("data", data);
        }else if("PMIDS".equals(contentType)) {
            Result tmp = GetAbstractUtil.getAbstract(content);
            if(tmp.isSuccess()) {
                requestData.put("contentType", contentType);
                requestData.put("data", tmp.getData());
            }else {
                return Result.ERROR();
            }
        }else {
            return Result.ERROR();
        }
        RestTemplate restTemplate = new RestTemplate();
        System.out.println("开始调用模型接口");
        try {
                ResponseEntity<String> responseEntity = restTemplate.postForEntity("http://localhost:5000/predict",requestData,String.class);
                String Body = responseEntity.getBody();
                JSONArray res = JSONObject.parseArray(Body);
                System.out.println("模型接口调用成功");
                return new Result<>(ResultCode.SUCCESS, res);
        } catch (Exception e) {
                System.out.println(e);
                return new Result(ResultCode.FAIL);
        }
    }
}

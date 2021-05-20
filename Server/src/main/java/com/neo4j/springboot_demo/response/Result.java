package com.neo4j.springboot_demo.response;

import lombok.Data;

import java.io.Serializable;

@Data
public class Result<T> implements Serializable {
    private static final long serialVersionUID = -1802122468331526708L;
    private boolean success;
    private Integer code;
    private String message;
    private T data;

    public Result(ResultCode code) {
        this.success = code.success;
        this.code = code.code;
        this.message = code.message;
    }

    public Result(ResultCode code, T data) {
        this.success = code.success;
        this.code = code.code;
        this.message = code.message;
        this.data = data;
    }

    public Result(Integer code, String message, boolean success) {
        this.code = code;
        this.message = message;
        this.success = success;
    }

    public static <T>Result<T> SUCCESS() {
        return new Result<T>(ResultCode.SUCCESS);
    }

    public static <T>Result<T> FAIL() {
        return new Result<T>(ResultCode.FAIL);
    }

    public static <T>Result<T> ERROR() {
        return new Result<T>(ResultCode.SERVER_ERROR);
    }

    public static <T>Result<T> NOT_EXIST() {
        return new Result<T>(ResultCode.DATA_NOT_EXIST);
    }

}

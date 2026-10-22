// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Two jobs:
// 1. cuvs-sys (`links = "cuvs"`) reports the directory of the libcuvs_c it found. Embed it as an
//    rpath so the extension loads that local build without LD_LIBRARY_PATH or LD_PRELOAD;
//    libcuvs_c's own rpath then finds its librmm/libraft/... from the conda env it was built in.
// 2. Compile the small row-normalisation CUDA kernel (src/normalize.cu, used by
//    LANCE_CUVS_COSINE_AS_L2=normalize) with nvcc into a static library.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-env-changed=DEP_CUVS_LIB");
    if let Ok(lib_dir) = std::env::var("DEP_CUVS_LIB") {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }

    println!("cargo:rerun-if-changed=src/normalize.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let cuda = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = format!("{cuda}/bin/nvcc");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("normalize.o");
    let status = Command::new(&nvcc)
        .args([
            "-c", "src/normalize.cu", "-o", obj.to_str().unwrap(), "-O3", "-cudart", "shared",
            "-Xcompiler", "-fPIC",
            "-gencode", "arch=compute_80,code=sm_80",
            "-gencode", "arch=compute_90,code=sm_90",
            "-gencode", "arch=compute_90,code=compute_90",
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
    assert!(status.success(), "nvcc failed");
    let lib = out.join("libnormalize.a");
    let status = Command::new("ar")
        .args(["crs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .expect("ar");
    assert!(status.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=normalize");
    for sub in ["lib64", "lib", "targets/x86_64-linux/lib"] {
        println!("cargo:rustc-link-search=native={cuda}/{sub}");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
}

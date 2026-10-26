plugins {
    id("com.android.library")
    kotlin("android")
    id("maven-publish")
}

group = "io.github.quangng2000"
version = "0.1.2"

android {
    namespace = "io.riftco.reuseaudio"
    compileSdk = 36
    ndkVersion = "27.2.12479018"
    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        externalNativeBuild { cmake { cppFlags += listOf("-O2", "-fno-fast-math", "-ffp-contract=on") } }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    publishing { singleVariant("release") }
    sourceSets.getByName("androidTest").assets.srcDir(layout.buildDirectory.dir("generated/parity-assets"))
    externalNativeBuild {
        cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" }
    }
}

val hostDspDirectory = layout.buildDirectory.dir("host-dsp")
val buildHostDsp by tasks.registering(Exec::class) {
    inputs.dir("src/main/cpp")
    inputs.file(rootProject.file("tools/build-host-dsp.sh"))
    outputs.dir(hostDspDirectory)
    commandLine("bash", rootProject.file("tools/build-host-dsp.sh").absolutePath,
        file("src/main/cpp").absolutePath, hostDspDirectory.get().asFile.absolutePath,
        System.getProperty("java.home"))
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    // Same Java API with host JNI, for real CPU parity outside an emulator.
    testRuntimeOnly("com.microsoft.onnxruntime:onnxruntime:1.23.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

val apiSourcesJar by tasks.registering(Jar::class) {
    archiveClassifier.set("sources")
    from("src/main/kotlin")
    from("src/main/cpp") { into("native") }
    from("src/main/assets/reuse-audio/licenses") { into("licenses") }
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
}

val apiDocumentationJar by tasks.registering(Jar::class) {
    archiveClassifier.set("javadoc")
    // Kotlin API KDoc is supplied as readable source alongside the integration guide.
    // Central explicitly permits a documentation JAR containing a README.
    from(rootProject.file("README.md"))
    from("src/main/kotlin") { into("api") }
    from("src/main/assets/reuse-audio/licenses") { into("licenses") }
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
}

afterEvaluate {
    publishing.publications.create<MavenPublication>("release") {
        from(components["release"])
        artifactId = "reuse-audio-android"
        artifact(apiSourcesJar)
        artifact(apiDocumentationJar)
        pom {
            name.set("RE-USE Audio for Android")
            description.set("On-device offline speech enhancement for Android using the 30-layer FP32 NVIDIA NemotronLabs RE-USE model with ONNX Runtime.")
            url.set("https://github.com/quangng2000/reuse-audio")
            inceptionYear.set("2026")
            licenses {
                license {
                    name.set("OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)")
                    url.set("https://github.com/quangng2000/reuse-audio/blob/android-v${project.version}/LICENSE")
                    distribution.set("repo")
                }
                license {
                    name.set("BSD-3-Clause (PocketFFT)")
                    url.set("https://github.com/quangng2000/reuse-audio/blob/android-v${project.version}/android/reuse-audio/src/main/cpp/POCKETFFT-LICENSE.txt")
                    distribution.set("repo")
                }
                license {
                    name.set("MIT (original JNI adapter)")
                    url.set("https://github.com/quangng2000/reuse-audio/blob/android-v${project.version}/android/reuse-audio/src/main/assets/reuse-audio/licenses/REUSE-DSP-LICENSE.txt")
                    distribution.set("repo")
                }
            }
            developers {
                developer {
                    id.set("quangng2000")
                    name.set("quangng2000")
                    url.set("https://github.com/quangng2000")
                }
            }
            scm {
                connection.set("scm:git:https://github.com/quangng2000/reuse-audio.git")
                developerConnection.set("scm:git:ssh://git@github.com/quangng2000/reuse-audio.git")
                url.set("https://github.com/quangng2000/reuse-audio")
                tag.set("android-v${project.version}")
            }
        }
    }
}

publishing.repositories.maven {
    name = "CentralStaging"
    url = uri(layout.buildDirectory.dir("central-repository"))
}

tasks.withType<Test>().configureEach {
    dependsOn(buildHostDsp)
    systemProperty("java.library.path", hostDspDirectory.get().asFile.absolutePath)
    listOf("REUSE_PARITY_DIR", "REUSE_MODEL_FILE", "REUSE_MODEL_MANIFEST").forEach { name ->
        providers.environmentVariable(name).orNull?.let { systemProperty(name, it) }
    }
}

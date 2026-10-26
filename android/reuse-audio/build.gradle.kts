plugins {
    id("com.android.library")
    kotlin("android")
    id("maven-publish")
}

group = "io.github.quangng2000"
version = "0.1.0"

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
    publishing { singleVariant("release") { withSourcesJar() } }
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

afterEvaluate {
    publishing.publications.create<MavenPublication>("release") {
        from(components["release"])
        artifactId = "reuse-audio-android"
    }
}

tasks.withType<Test>().configureEach {
    dependsOn(buildHostDsp)
    systemProperty("java.library.path", hostDspDirectory.get().asFile.absolutePath)
    listOf("REUSE_PARITY_DIR", "REUSE_MODEL_FILE", "REUSE_MODEL_MANIFEST").forEach { name ->
        providers.environmentVariable(name).orNull?.let { systemProperty(name, it) }
    }
}

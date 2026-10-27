plugins { id("com.android.application"); kotlin("android") }

val streamingSource = providers.gradleProperty("streamingSdkSource").orElse("published").get()
require(streamingSource in setOf("published", "local"))
val streamingVariant = providers.gradleProperty("streamingVariant").orElse(streamingSource).get()
require(streamingVariant.matches(Regex("[A-Za-z0-9_.-]+")))
val nativeFrequencyScan = providers.gradleProperty("streamingNativeScan").orNull == "true"
val nativeVulkan = providers.gradleProperty("streamingVulkan").orNull == "true"
require(!nativeFrequencyScan || streamingSource == "local")
require(!nativeVulkan || (streamingSource == "local" && nativeFrequencyScan))

android {
    namespace = "io.riftco.reusestreaming.benchmark"
    compileSdk = 36
    defaultConfig {
        applicationId = "io.riftco.reusestreaming.benchmark"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.3-benchmark"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "STREAMING_SDK_SOURCE", "\"$streamingSource\"")
        buildConfigField("String", "STREAMING_VARIANT", "\"$streamingVariant\"")
        buildConfigField("boolean", "NATIVE_FREQUENCY_SCAN", nativeFrequencyScan.toString())
        ndk { abiFilters += "arm64-v8a" }
    }
    buildTypes {
        release {
            isDebuggable = false
            isMinifyEnabled = false
            isShrinkResources = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    testBuildType = "release"
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { buildConfig = true }
    sourceSets.getByName("main").assets.srcDir(layout.buildDirectory.dir("benchmark-assets"))
    if (nativeFrequencyScan) {
        sourceSets.getByName("main").assets.srcDir(layout.buildDirectory.dir("benchmark-native-assets"))
        sourceSets.getByName("main").jniLibs.srcDir(layout.buildDirectory.dir("benchmark-native-libs"))
        packaging.jniLibs.useLegacyPackaging = true
        packaging.jniLibs.keepDebugSymbols += "**/libreuse_streaming_scan.so"
        packaging.jniLibs.keepDebugSymbols += "**/libreuse_streaming_native_v2.so"
        packaging.jniLibs.keepDebugSymbols += "**/libreuse_streaming_inline_v2.so"
        packaging.jniLibs.keepDebugSymbols += "**/libreuse_streaming_stable_v2.so"
    }
    if (nativeVulkan) {
        sourceSets.getByName("main").assets.srcDir(layout.buildDirectory.dir("benchmark-vulkan-assets"))
        sourceSets.getByName("main").jniLibs.srcDir(layout.buildDirectory.dir("benchmark-vulkan-libs"))
        sourceSets.getByName("androidTest").java.srcDir("src/vulkanAndroidTest/kotlin")
        packaging.jniLibs.keepDebugSymbols += "**/libreuse_streaming_vulkan.so"
    }
}

dependencies {
    implementation("io.github.quangng2000:reuse-streaming-android:0.1.3")
    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
}

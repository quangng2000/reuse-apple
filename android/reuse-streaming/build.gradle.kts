import org.gradle.api.tasks.bundling.Jar
import org.gradle.api.tasks.testing.Test
import org.gradle.api.publish.maven.MavenPublication

plugins { id("com.android.library"); kotlin("android"); id("maven-publish") }

group = "io.github.quangng2000"
version = "0.1.2"

android {
    namespace = "io.riftco.reusestreaming"
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
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
}

val hostDspDirectory = layout.buildDirectory.dir("host-dsp")
val buildHostDsp by tasks.registering(Exec::class) {
    inputs.dir("src/main/cpp")
    inputs.file("tools/build-host-dsp.sh")
    outputs.dir(hostDspDirectory)
    commandLine("bash", file("tools/build-host-dsp.sh").absolutePath,
        file("src/main/cpp").absolutePath, hostDspDirectory.get().asFile.absolutePath,
        System.getProperty("java.home"))
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    testRuntimeOnly("com.microsoft.onnxruntime:onnxruntime:1.23.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

val apiSourcesJar by tasks.registering(Jar::class) {
    archiveClassifier.set("sources")
    from("src/main/kotlin")
    from("src/main/cpp") { into("native") }
    from("src/main/assets/reuse-streaming/licenses") { into("licenses") }
    isPreserveFileTimestamps = false; isReproducibleFileOrder = true
}
val apiDocumentationJar by tasks.registering(Jar::class) {
    archiveClassifier.set("javadoc")
    from("README.md")
    from("src/main/kotlin") { into("api") }
    from("src/main/assets/reuse-streaming/licenses") { into("licenses") }
    isPreserveFileTimestamps = false; isReproducibleFileOrder = true
}
afterEvaluate {
    publishing.publications.create<MavenPublication>("release") {
        from(components["release"])
        artifactId = "reuse-streaming-android"
        artifact(apiSourcesJar); artifact(apiDocumentationJar)
        pom {
            name.set("RE-USE Streaming for Android (Research)")
            description.set("Stateful on-device FP32 Real-time RE-USE streaming for Android. NVIDIA One-Way Noncommercial License (NSCLv1); noncommercial research and education only. Separate from the commercial offline SDK.")
            url.set("https://github.com/quangng2000/reuse-audio")
            inceptionYear.set("2026")
            licenses {
                license {
                    name.set("NVIDIA One-Way Noncommercial License (NSCLv1)")
                    url.set("https://github.com/quangng2000/reuse-audio/blob/android-v${project.version}/android/reuse-streaming/src/main/assets/reuse-streaming/licenses/LICENSE.txt")
                    distribution.set("repo")
                }
                license { name.set("BSD 3-Clause License (PocketFFT)"); url.set("https://opensource.org/license/bsd-3-clause"); distribution.set("repo") }
                license { name.set("MIT License (JNI adapter and ONNX Runtime)"); url.set("https://opensource.org/license/mit"); distribution.set("repo") }
            }
            developers { developer { id.set("quangng2000"); name.set("Quang Nguyen"); url.set("https://github.com/quangng2000") } }
            scm {
                connection.set("scm:git:https://github.com/quangng2000/reuse-audio.git")
                developerConnection.set("scm:git:ssh://git@github.com/quangng2000/reuse-audio.git")
                url.set("https://github.com/quangng2000/reuse-audio")
                tag.set("android-v${project.version}")
            }
        }
    }
}
publishing.repositories.maven { name = "CentralStaging"; url = uri(layout.buildDirectory.dir("central-repository")) }
tasks.withType<Test>().configureEach {
    dependsOn(buildHostDsp)
    systemProperty("java.library.path", hostDspDirectory.get().asFile.absolutePath)
    listOf("REUSE_STREAMING_MODEL_FILE", "REUSE_STREAMING_MODEL_MANIFEST", "REUSE_STREAMING_PARITY_DIR").forEach { name ->
        System.getenv(name)?.let { systemProperty(name, it) }
    }
}

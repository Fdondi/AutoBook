plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.autoreader"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.example.autoreader"
        minSdk = 34
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        externalNativeBuild {
            cmake {
                // llama.cpp unoptimized is unusably slow, so even debug builds compile it for release.
                arguments += listOf("-DCMAKE_BUILD_TYPE=Release", "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON")
            }
        }
    }

    ndkVersion = "28.2.13676358"
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
}

/**
 * The reader screen is the desktop web reader (src/tbook/reader/static) in a WebView,
 * with PDF.js and KaTeX from the npm packages pinned in tools/package.json
 * (`npm install` in tools/ first), copied into the APK's assets under web/. The
 * reader's prompts (src/tbook/prompts) go under prompts/.
 */
abstract class WebAssetsTask : DefaultTask() {
    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val readerDir: DirectoryProperty

    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val promptsDir: DirectoryProperty

    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val pdfjsDir: DirectoryProperty

    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val katexDir: DirectoryProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @get:Inject
    abstract val fs: FileSystemOperations

    @TaskAction
    fun copy() {
        fs.sync {
            into(outputDir)
            from(readerDir) { into("web/static") }
            from(pdfjsDir) {
                include("legacy/build/pdf.mjs", "legacy/build/pdf.worker.mjs", "legacy/web/pdf_viewer.mjs",
                        "legacy/web/pdf_viewer.css", "legacy/web/images/**", "cmaps/**", "standard_fonts/**",
                        "wasm/**", "iccs/**")
                into("web/vendor/pdfjs")
            }
            from(katexDir) {
                include("katex.mjs", "katex.min.css", "fonts/**")
                into("web/vendor/katex")
            }
            from(promptsDir) {
                include("reader_answer.md", "leak_semantic.md", "reader_grade.md")
                into("prompts")
            }
        }
    }
}

val repo = rootProject.layout.projectDirectory.dir("..")
val webAssets = tasks.register<WebAssetsTask>("webAssets") {
    readerDir.set(repo.dir("src/tbook/reader/static"))
    promptsDir.set(repo.dir("src/tbook/prompts"))
    pdfjsDir.set(repo.dir("tools/node_modules/pdfjs-dist"))
    katexDir.set(repo.dir("tools/node_modules/katex/dist"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(webAssets, WebAssetsTask::outputDir)
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.webkit)
    testImplementation(libs.junit)
    testImplementation(libs.sqlite.jdbc)     // the pipeline's parity tests read the exported book
    testImplementation(libs.json)            // org.json: Android's copy is a stub in JVM tests
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}

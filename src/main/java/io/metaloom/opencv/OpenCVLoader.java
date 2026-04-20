package io.metaloom.opencv;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.SymbolLookup;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Loads the native opencv_ffm shared library and provides the symbol lookup
 * used by {@link NativeBindings}.
 */
public final class OpenCVLoader {

    private static volatile SymbolLookup lookup;
    private static volatile Arena arena;
    private static volatile boolean loaded = false;

    private OpenCVLoader() {
    }

    /**
     * Load the native library from the given directory path.
     * The directory must contain {@code libopencv_ffm.so} (Linux).
     */
    public static synchronized void load(Path libraryDir) {
        if (loaded) {
            return;
        }
        Path libPath = libraryDir.resolve(System.mapLibraryName("opencv_ffm"));
        arena = Arena.ofAuto();
        lookup = SymbolLookup.libraryLookup(libPath, arena);
        loaded = true;
    }

    /**
     * Load using the default library search path. The library is resolved in
     * order:
     * <ol>
     *   <li>Directories listed in {@code LD_LIBRARY_PATH}</li>
     *   <li>Extraction from JAR resources ({@code /native/linux/})</li>
     *   <li>{@code System.loadLibrary} fallback</li>
     * </ol>
     */
    public static synchronized void load() {
        if (loaded) {
            return;
        }
        String libName = "opencv_ffm";
        String libFileName = System.mapLibraryName(libName);
        arena = Arena.ofAuto();

        // First try: explicit path from LD_LIBRARY_PATH
        String ldPath = System.getenv("LD_LIBRARY_PATH");
        if (ldPath != null) {
            for (String dir : ldPath.split(":")) {
                Path candidate = Path.of(dir, libFileName);
                if (candidate.toFile().exists()) {
                    lookup = SymbolLookup.libraryLookup(candidate, arena);
                    loaded = true;
                    return;
                }
            }
        }

        // Second try: extract from JAR resources
        try {
            File extracted = extractFromResources(libFileName);
            if (extracted != null) {
                System.load(extracted.getAbsolutePath());
                lookup = SymbolLookup.loaderLookup();
                loaded = true;
                return;
            }
        } catch (Exception e) {
            // Fall through to System.loadLibrary
        }

        // Fallback: use System.loadLibrary and loaderLookup
        System.loadLibrary(libName);
        lookup = SymbolLookup.loaderLookup();
        loaded = true;
    }

    private static File extractFromResources(String libFileName) {
        String os = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);
        String resourcePath;
        if (os.contains("linux")) {
            resourcePath = "/native/linux/" + libFileName;
        } else if (os.contains("mac")) {
            resourcePath = "/native/macosx/" + libFileName;
        } else {
            return null;
        }

        try (InputStream in = OpenCVLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                return null;
            }
            File tempFile = File.createTempFile("opencv_ffm", ".so");
            tempFile.deleteOnExit();
            try (OutputStream out = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = in.read(buffer)) != -1) {
                    out.write(buffer, 0, read);
                }
            }
            return tempFile;
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Returns the symbol lookup for the loaded native library.
     *
     * @throws IllegalStateException if the library has not been loaded
     */
    public static SymbolLookup lookup() {
        if (!loaded) {
            throw new IllegalStateException(
                    "OpenCV native library not loaded. Call OpenCVLoader.load() first.");
        }
        return lookup;
    }

    public static boolean isLoaded() {
        return loaded;
    }
}

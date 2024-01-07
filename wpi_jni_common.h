#define WPI_JNI_MAKEJARRAY(T, F)                                               \
  inline T##Array MakeJ##F##Array(JNIEnv *env, T *data, size_t size) {         \
    T##Array jarr = env->New##F##Array(size);                                  \
    if (!jarr) {                                                               \
      return nullptr;                                                          \
    }                                                                          \
    env->Set##F##ArrayRegion(jarr, 0, size, data);                             \
    return jarr;                                                               \
  }

WPI_JNI_MAKEJARRAY(jboolean, Boolean)
WPI_JNI_MAKEJARRAY(jbyte, Byte)
WPI_JNI_MAKEJARRAY(jshort, Short)
WPI_JNI_MAKEJARRAY(jlong, Long)
WPI_JNI_MAKEJARRAY(jfloat, Float)
WPI_JNI_MAKEJARRAY(jdouble, Double)

#undef WPI_JNI_MAKEJARRAY

/**
 * Finds a class and keeps it as a global reference.
 *
 * Use with caution, as the destructor does NOT call DeleteGlobalRef due to
 * potential shutdown issues with doing so.
 */
class JClass {
public:
  JClass() = default;

  JClass(JNIEnv *env, const char *name) {
    jclass local = env->FindClass(name);
    if (!local) {
      return;
    }
    m_cls = static_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
  }

  void free(JNIEnv *env) {
    if (m_cls) {
      env->DeleteGlobalRef(m_cls);
    }
    m_cls = nullptr;
  }

  explicit operator bool() const { return m_cls; }

  operator jclass() const { return m_cls; }

protected:
  jclass m_cls = nullptr;
};

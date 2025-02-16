int float_near_abs_eps_array(const float *a, const float *b, float eps,
                         unsigned len)
{
    unsigned i;

    for (i = 0; i < len; i++) {
        if (!float_near_abs_eps(a[i], b[i], eps))
            return 0;
    }
    return 1;
}
static int randomize(uint8_t *buf, int len)
{
#if CONFIG_GCRYPT
    gcry_randomize(buf, len, GCRY_VERY_STRONG_RANDOM);
    return 0;
#elif CONFIG_OPENSSL
    if (RAND_bytes(buf, len))
        return 0;
#else
    return AVERROR(ENOSYS);
#endif
    return AVERROR(EINVAL);
}
static void check_func(int value, int line, const char *msg, ...)
{
    if (!value) {
        va_list ap;
        va_start(ap, msg);
        printf("%d: ", line);
        vprintf(msg, ap);
        printf("\n");
        check_faults++;
        va_end(ap);
    }
}

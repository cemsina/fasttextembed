#include "loader.h"
#include "config.h"
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int fte_weights_open(const char *path, fte_weights *w) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return -1;

    const fte_header *h = (const fte_header *)m;
    if (h->magic != FTE_MAGIC || h->version != FTE_VERSION) {
        munmap(m, (size_t)st.st_size);
        return -2;
    }
    if (h->hidden != FTE_HIDDEN || h->layers != FTE_LAYERS || h->heads != FTE_HEADS ||
        h->intermediate != FTE_INTERMEDIATE || h->vocab != FTE_VOCAB) {
        munmap(m, (size_t)st.st_size);
        return -3;
    }
    w->map = m;
    w->map_size = (size_t)st.st_size;
    w->hdr = h;
    w->table = (const fte_tensor_entry *)((const char *)m + h->table_offset);
    return 0;
}

void fte_weights_close(fte_weights *w) {
    if (w->map) { munmap(w->map, w->map_size); w->map = NULL; }
}

const void *fte_weight(const fte_weights *w, const char *name) {
    for (uint32_t i = 0; i < w->hdr->n_tensors; i++)
        if (strncmp(w->table[i].name, name, FTE_NAME_MAX) == 0)
            return (const void *)((const char *)w->map + w->table[i].offset);
    return NULL;
}

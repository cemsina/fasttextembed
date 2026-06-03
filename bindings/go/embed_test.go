package fasttextembed

import "testing"

func TestEmbed(t *testing.T) {
	m, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	v := m.EmbedOne("hello world")
	if len(v) != Dim {
		t.Fatalf("dim %d", len(v))
	}
	t.Logf("dim=%d first5=%v", len(v), v[:5])
	b := m.Embed([]string{"hello world", "the quick brown fox"})
	t.Logf("batch=%dx%d", len(b), len(b[0]))
}

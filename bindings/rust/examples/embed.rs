use fasttextembed::TextEmbedding;

fn main() {
    let model = TextEmbedding::new().expect("init");
    let v = model.embed_one("hello world");
    println!("dim={} first5={:?}", v.len(), &v[..5]);
    let b = model.embed(&["hello world", "the quick brown fox"]);
    println!("batch={}x{}", b.len(), b[0].len());
}

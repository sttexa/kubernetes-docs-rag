from __future__ import annotations

import unittest

from app.services.chunker import MAX_CHUNK_CHARS, chunk_html


class ChunkerTests(unittest.TestCase):
    def test_chunk_html_includes_h4_sections(self) -> None:
        html = """
        <html>
          <body>
            <main>
              <article>
                <h1>Deployment</h1>
                <h2>Overview</h2>
                <p>Deployments manage replicated Pods and rollout updates safely across a cluster.</p>
                <p>They provide declarative updates and make it easier to reason about desired state.</p>
                <h4>Progressing Deployments</h4>
                <p>Progressing status reports whether the rollout is advancing and available.</p>
                <p>Controllers update conditions during rollout and completion to help operators debug.</p>
              </article>
            </main>
          </body>
        </html>
        """

        chunks = chunk_html(html, "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/")

        self.assertTrue(chunks)
        self.assertTrue(any("Progressing Deployments" in chunk.heading_path for chunk in chunks))

    def test_chunk_html_splits_large_sections(self) -> None:
        paragraph = " ".join(["deployment"] * 220)
        html = f"""
        <html>
          <body>
            <main>
              <article>
                <h1>Deployment</h1>
                <h2>Large Section</h2>
                <p>{paragraph}</p>
                <p>{paragraph}</p>
                <p>{paragraph}</p>
              </article>
            </main>
          </body>
        </html>
        """

        chunks = chunk_html(html, "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/")

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk.text) <= MAX_CHUNK_CHARS + 50 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()

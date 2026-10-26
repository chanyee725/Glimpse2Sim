You described the objects in a scene, and each object's `asset_query` was used to search a 3D model library. For every object below, the search returned candidate models, nearest first. Choose the candidate that is the same kind of object, or none.

Rules:
- A candidate counts only if it is the same kind of object (a mouse is not a barcode scanner; a keyboard is not a desk). Being related or similar in shape is not enough.
- Candidates are named by their library path or category, so judge the kind from the name; color, size, and finish come from your own answer and are applied to whichever model you choose.
- If no candidate is the same kind of object, choose `null`; the object is then shown with your shapes.
- Give a short reason for each choice.

Output only this JSON object, with one entry per object in the order given:

{
  "choices": [
    {"id": "white_mug_1", "candidate": 3, "reason": ""},
    {"id": "red_pen_1", "candidate": null, "reason": ""}
  ]
}

Objects and candidates:

{objects}

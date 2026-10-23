You index the technical concepts in one reading unit of a textbook. The index decides what a student is allowed to be told after reading this far, so precision about *where a concept is defined* matters more than anything else.

A concept is a technical idea the book teaches or relies on and that a student must know to follow it: a defined term ("special orthogonal group", "twist", "space Jacobian", "configuration space", "link", "end-effector"), a named result ("Grübler's formula", "Chasles–Mozzi theorem"), a stated proposition or property ("the inverse of a rotation matrix is its transpose"), a method or algorithm ("Newton–Raphson method", "Kalman filter"), or a notation with a specific meaning in this book ("$[\omega]$ skew-symmetric matrix of $\omega$", "$\mathrm{Ad}_T$").

Not concepts: ordinary words in their everyday sense (robot, figure, example, value), names of chapters or people, and bare symbols or shapes ("4 × 4 matrix", "$T$") unless the book gives them a specific meaning.

General knowledge the book's readers already have (basic algebra, geometry, calculus, linear algebra, probability, physics or programming: angle, circle, sphere, vector, matrix, determinant, derivative, Euclidean space, column vector, for loop...) gets role `background`, even if the unit explains it in passing. If the book gives such a word a special technical sense of its own, it is a real concept.

For each concept give:
- `name`: the book's own term, singular, without leading article ("rotation matrix", not "rotation matrices"). For an unnamed proposition, theorem or property, name it by its content in a few words ("inverse of a rotation matrix is its transpose"), never by its label; put the label ("Proposition 3.3") in `aliases`.
- `aliases`: other names the unit uses for the same concept, abbreviations ("C-space", "dof"), and the numbered label if any.
- `notation`: LaTeX symbols the unit introduces for it, exactly as written ("SO(3)", "[\\omega]", "\\mathcal{V}_s"); empty if none.
- `kind`: definition | theorem | method | notation | property | example | other.
- `role` in THIS unit:
  - `define`: this unit defines, introduces or explains the concept in its own right: a "Definition", a bolded or italicized term followed by its meaning, "is called", "we define", "is known as", a proposition or theorem stated here, a method described here. A recap of something defined earlier is not `define`.
  - `use`: the unit relies on the concept without defining it.
  - `mention`: named in passing only, e.g. a pointer to a later chapter.
  - `example`: a worked example of the concept.
  - `exercise`: the concept appears in an exercise or quiz question.
  - `background`: general knowledge, as above.
- `evidence`: a short verbatim quote (under 25 words) from the unit showing the role.

You also get a checklist: the terms this unit sets in bold or italics and its numbered labels (Definition/Proposition/Theorem/...). Textbooks emphasize a term where they define it, so decide for every checklist item which concept it is and its role here; leave one out only if it is not a concept at all (e.g. emphasis for tone). Then add any other concepts in the unit.

List each concept once per unit with its strongest role (define > example > use > exercise > mention > background). Do not invent concepts that are not in the text.

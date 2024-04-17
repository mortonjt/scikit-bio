# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------
from skbio.sequence import Protein
from skbio.embedding._embedding import SequenceEmbedding
from skbio.util import get_data_path
from pathlib import Path
from typing import List


class ProteinEmbedding(SequenceEmbedding):
    """Stores the embeddings of the protein sequence.

    Parameters
    ----------
    sequence : str, Sequence, or 1D np.ndarray
        Characters representing the protein sequence itself.
    embedding : np.ndarray
        The embedding of the protein sequence. Row vectors correspond to
        the latent residues coordinates.
    clip_head : bool, optional
        If ``True``, then the first row of the embedding will be removed.
        Some language models specify start tokens, and this parameter can
        be used to account for this.
    clip_end : bool, optional
        If ``True``, then the last row of the embedding will be removed.
        Some language models specify end tokens, and this parameter can
        be used to account for this.

    See Also
    --------
    Protein

    """

    default_write_format = "embed"

    def __init__(
        self, embedding, sequence: str, clip_head=False, clip_tail=False, **kwargs
    ):
        if clip_head:
            embedding = embedding[1:]
        if clip_tail:
            embedding = embedding[:-1]

        if isinstance(sequence, Protein):
            sequence = str(sequence)

        # make sure that the embedding has the same length as the sequence
        sequence_len = len(sequence)
        if embedding.shape[0] != sequence_len:
            raise ValueError(
                f"The embedding ({embedding.shape[0]}) must have the "
                f"same length as the sequence ({len(sequence)})."
            )

        super(ProteinEmbedding, self).__init__(
            embedding=embedding, sequence=sequence, **kwargs
        )

    def __str__(self):
        return str(self._ids)

    def __repr__(self):
        """
        Return a string representation of the ProteinEmbedding object.

        Returns
        -------
        str
            A string representation of the ProteinEmbedding object.

        See Also
        --------
        Protein
        """
        seq = Protein(str(self._ids))

        rstr = repr(seq)
        rstr = rstr.replace("Protein", "ProteinEmbedding")
        n_indent = 4  # see Sequence.__repr__
        indent = " " * n_indent
        rstr = rstr.replace(
            "has gaps",
            f"embedding dimension: {self.embedding.shape[1]}\n{indent}has gaps",
        )
        return rstr


class ProteinVector(SequenceVector):
    """ A vector representation of the protein sequence.

    Parameters
    ----------
    sequence : str, Sequence, or 1D np.ndarray
        Characters representing the protein sequence itself.
    vector : np.ndarray
        The vector representation of the protein sequence.

    See Also
    --------
    Protein

    """

    def __init__(
        self, vector, sequence: str, **kwargs
    ):
        super(ProteinVector, self).__init__(
            embedding=vector, sequence=sequence,  **kwargs
        )

    def __str__(self):
        return str(self._ids)

    def __repr__(self):
        """
        Return a string representation of the ProteinVector object.

        Returns
        -------
        str
            A string representation of the ProteinEmbedding object.

        See Also
        --------
        Protein
        """
        seq = Protein(str(self._ids))

        rstr = repr(seq)
        rstr = rstr.replace("Protein", "ProteinVector")
        n_indent = 4  # see Sequence.__repr__
        indent = " " * n_indent
        rstr = rstr.replace(
            "has gaps",
            f"vector dimension: {self.embedding.shape[1]}\n{indent}has gaps",
        )
        return rstr

    @staticmethod
    def to_distance_matrix(protein_vectors):
        """
        Convert a ProteinVector object to a DistanceMatrix object.

        Parameters
        ----------
        protein_vectors : iterable of ProteinVector objects
            An iterable of ProteinVector objects.

        Returns
        -------
        DistanceMatrix
            A DistanceMatrix object.

        See Also
        --------
        DistanceMatrix
        """
        pass

    @staticmethod
    def to_ordination(protein_vectors):
        """
        Convert a list of ProteinVector objects to an Ordination object.

        Parameters
        ----------
        protein_vectors : iterable of ProteinVector objects
            An iterable of ProteinVector objects.

        Returns
        -------
        Ordination
            An Ordination object.

        See Also
        --------
        Ordination
        """
        pass

    @staticmethod
    def to_dataframe(protein_vectors):
        """
        Convert a list of ProteinVector objects to a pandas DataFrame.

        Parameters
        ----------
        protein_vectors : iterable of ProteinVector objects
            An iterable of ProteinVector objects.

        Returns
        -------
        DataFrame
            A pandas DataFrame.

        See Also
        --------
        DataFrame
        """
        pass

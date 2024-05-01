# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------
from skbio.sequence import Protein
from skbio.embedding._embedding import SequenceEmbedding
from skbio.stats.ordination import OrdinationResults
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.util import get_data_path
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List


class ProteinEmbedding(SequenceEmbedding):
    r"""Stores the embeddings of the protein sequence.

    Parameters
    ----------
    sequence : str, Protein, or 1D np.ndarray
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
        self, embedding, sequence, clip_head=False, clip_tail=False, **kwargs
    ):
        if clip_head:
            embedding = embedding[1:]
        if clip_tail:
            embedding = embedding[:-1]

        if isinstance(sequence, Protein):
            sequence = str(sequence)

        if " " in sequence:
            sequence = sequence.replace(" ", "")

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


example_protein_embedding = ProteinEmbedding(
    np.random.randn(62, 1024),
    'IGKEEIQQRLAQFVDHWKELKQLAAARGQRLEESLEYQQFVANVEEEEAWINEKMTLVASED')


class ProteinVector(SequenceEmbedding):
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

        if isinstance(sequence, Protein):
            sequence = str(sequence)

        if " " in sequence:
            sequence = sequence.replace(" ", "")

        super(ProteinVector, self).__init__(
            embedding=vector, sequence=sequence,  **kwargs
        )

    @property
    def vector(self):
        return self.embedding

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
    def to_numpy(protein_vectors : List[ProteinVector]):
        lens = [len(pv.vector) for pv in protein_vectors]
        if not all(l == lens[0] for l in lens):
            raise ValueError("All vectors must have the same length.")
        data = np.vstack([pv.vector for pv in protein_vectors])
        return data

    @staticmethod
    def to_distance_matrix(protein_vectors : List[ProteinVector],
                           metric='euclidean'):
        """
        Convert a ProteinVector object to a DistanceMatrix object.

        Parameters
        ----------
        protein_vectors : iterable of ProteinVector objects
            An iterable of ProteinVector objects.
        metric : str, optional
            The distance metric to use. Must be a valid metric for
            `scipy.spatial.distance.pdist`.

        Returns
        -------
        DistanceMatrix
            A DistanceMatrix object.

        See Also
        --------
        DistanceMatrix
        """
        data = ProteinVector.to_numpy(protein_vectors)
        ids = [str(pv) for pv in protein_vectors]
        dm = squareform(pdist(data, metric))
        return DistanceMatrix(dm, ids=ids)

    @staticmethod
    def to_ordination(protein_vectors : List[ProteinVector]):
        """
        Convert a list of ProteinVector objects to an Ordination object.

        Parameters
        ----------
        protein_vectors : iterable of ProteinVector objects
            An iterable of ProteinVector objects.

        Returns
        -------
        OrdinationResults
            An Ordination object.

        See Also
        --------
        OrdinationResults
        """
        data = ProteinVector.to_numpy(protein_vectors)
        u, s, v = np.linalg.svd(data)
        eigvals = s ** 2
        ordr = OrdinationResults(
            short_method_name = 'ProteinVectors',
            long_method_name = 'ProteinVectors',
            eigvals = eigvals,
            proportion_explained = eigvals / eigvals.sum(),
            samples=pd.DataFrame(u * s, index=[str(pv) for pv in protein_vectors]),
            features=pd.DataFrame(v.T * s, index=range(data.shape[1])),
        )
        return ordr

    @staticmethod
    def to_dataframe(protein_vectors : List[ProteinVector]):
        """
        Convert a list of ProteinVector objects to a pandas DataFrame.

        Parameters
        ----------
        protein_vectors : iterable of ProteinVector objects
            An iterable of ProteinVector objects.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame.

        See Also
        --------
        pd.DataFrame
        """
        data = ProteinVector.to_numpy(protein_vectors)
        df = pd.DataFrame(data, index=[str(pv) for pv in protein_vectors])
        return df
